
from orchard.workflow_utils import load_rxns
from orchard.workflow_utils import SAVE_ROOT
import os, yaml, copy
import dftd4.pyscf as pyd4
import numpy as np
import numpy

from pyscf import scf, gto

from argparse import ArgumentParser

from ciderpress.models.jax_pw6b95 import pw6b95_train, pw8b95, \
    pw11b95, pw12b95, pw14b95, pw13b95, rpw12b95, \
    build_xcfunc_and_param_grad, PW6B95_DEFAULT_PARAMS, \
    PW8B95_DEFAULT_PARAMS, PW11B95_DEFAULT_PARAMS, \
    PW12B95_DEFAULT_PARAMS, PW14B95_DEFAULT_PARAMS, PW13B95_DEFAULT_PARAMS
from ciderpress.analyzers import ElectronAnalyzer as Analyzer


def get_base_energy(analyzer, d4func=None):
    restricted = True if analyzer.dm.ndim == 2 else False
    analyzer.mol.build()
    print(analyzer.mol.charge, analyzer.mol.spin)
    if restricted:
        calc = scf.RHF(analyzer.mol).density_fit()
    else:
        calc = scf.UHF(analyzer.mol).density_fit()
    calc.with_df.auxbasis = 'def2-universal-jkfit'
    dm = analyzer.dm
    h1e = calc.get_hcore()
    e_base = calc.energy_nuc()
    if restricted:
        e_base += numpy.einsum('ij,ji->', h1e, dm).real
    else:
        e_base += numpy.einsum('ij,xji->', h1e, dm).real
    #e_base+= np.sum(np.dot(analyzer.get('ha_energy_density'), analyzer.grids.weights))
    if restricted:
        e_base += 0.5 * numpy.einsum('ij,ji->', calc.get_j(dm=dm), dm).real
    else:
        e_base += 0.5 * numpy.einsum('ij,xji->', calc.get_j(dm=dm).sum(axis=0), dm).real
    if d4func is not None:
        e_base += pyd4.DFTD4Dispersion(
            calc.mol, xc=d4func.upper().replace(" ", "")
        ).kernel()[0]
    return e_base

def load_molecular_data(basis, functional, mol_id, d4_functional=None):
    print('MOL LOAD', mol_id)
    #d = os.path.join(SAVE_ROOT, 'KS', functional, basis, mol_id, 'analysis_L1.hdf5')
    d = os.path.join(SAVE_ROOT, 'KS', functional, basis, mol_id, 'analysis_L3.hdf5')
    analyzer = Analyzer.load(d)
    analyzer.set('restricted', analyzer.dm.ndim == 2)
    analyzer.set('e_base', get_base_energy(analyzer, d4_functional))
    analyzer.set('grids', analyzer.grids)
    analyzer.set('mol', analyzer.mol)
    eb, exc, etot = analyzer.get('e_base'), analyzer.get('exc_orig'), analyzer.get('e_tot_orig')
    if abs(eb+exc-etot) > 1e-3:
        print(eb, exc, etot, eb+exc-etot)
    return analyzer._data

def get_jax_inputs_dict(mol_data, mlfunc=None):
    inputs_dict = {}
    _aca = np.ascontiguousarray
    tol = 1e-9
    for mol_id, data in mol_data.items():
        print('MOL SETUP', mol_id)
        inp = {}
        inp['e_base'] = data['e_base']
        rho = data['rho_data']
        exx = data['ex_energy_density']
        if mlfunc is not None:
            LDA_FACTOR = - 3.0 / 4.0 * (3.0 / np.pi)**(1.0/3)
            feats = data['cider_descriptor_data']
            sel = mlfunc.desc_order[:5] if mlfunc.desc_version == 'd' else mlfunc.desc_order[:6]
            if data['restricted']:
                FX = mlfunc.get_F(feats[sel])
                xlda = LDA_FACTOR * rho[0]**(4.0/3)
                exx = FX * xlda
            else:
                FXa = mlfunc.get_F(feats[0][sel])
                FXb = mlfunc.get_F(feats[1][sel])
                xlda_a = 2**(1.0/3) * LDA_FACTOR * rho[0,0]**(4.0/3)
                xlda_b = 2**(1.0/3) * LDA_FACTOR * rho[1,0]**(4.0/3)
                exx = np.stack([FXa * xlda_a, FXb * xlda_b])
        if data['restricted']:
            cond = rho[0] > tol
            sigma = 0.25 * np.einsum('xg,xg->g', rho[1:4,cond], rho[1:4,cond])
            inp['xc_inputs'] = (
                _aca(0.5 * rho[0,cond]), _aca(0.5 * rho[0,cond]),
                sigma, sigma, sigma,
                _aca(0.5 * rho[5,cond]), _aca(0.5 * rho[5,cond]),
                _aca(0.5 * exx[cond]), _aca(0.5 * exx[cond]),
            )
        else:
            cond = rho[:,0].sum(axis=0) > tol
            grad_a = rho[0,1:4][:,cond]
            grad_b = rho[1,1:4][:,cond]
            sigma_aa = np.einsum('xg,xg->g', grad_a, grad_a)
            sigma_ab = np.einsum('xg,xg->g', grad_a, grad_b)
            sigma_bb = np.einsum('xg,xg->g', grad_b, grad_b)
            inp['xc_inputs'] = (
                _aca(rho[0,0,cond]), _aca(rho[1,0,cond]),
                sigma_aa, sigma_ab, sigma_bb,
                _aca(rho[0,5,cond]), _aca(rho[1,5,cond]),
                _aca(exx[0,cond]), _aca(exx[1,cond]),
            )
        inp['weights'] = _aca(data['grids'].weights[cond])
        inputs_dict[mol_id] = inp
    return inputs_dict


def compute_mol_preds_and_derivs(vpg_func, params, inputs_dict):
    predictions = {}
    for mol_id, data in inputs_dict.items():
        print('MOL-JAX', mol_id)
        exc, dexc = vpg_func(params, data['xc_inputs'], data["weights"])
        predictions[mol_id] = {"energy": data['e_base'] + exc, "grad": dexc}
    return predictions

def compute_rxn_preds_and_derivs(formulas, mol_predictions, pnames):
    ha_per_kcal = 0.001593601
    rxn_predictions = {}
    for rxn_id, rxn in formulas.items():
        #print('RXN', rxn_id)
        rp = {
            'e_pred' : 0,
            'de_pred' : {param : 0 for param in pnames},
            'e_ref' : ha_per_kcal * formulas[rxn_id]['energy'],
            'weight' : 1.0 / (formulas[rxn_id].get('noise_factor') or 1.0)**2,
        }
        for struct, count in zip(rxn['structs'], rxn['counts']):
            mp = mol_predictions[struct]
            rp['e_pred'] += count * mp['energy']
            for param in pnames:
                rp['de_pred'][param] += count * mp['grad'][param]
        rxn_predictions[rxn_id] = rp
    return rxn_predictions

def compute_loss_and_grad(rxn_predictions, pnames):
    loss = 0
    dloss = {param:0 for param in pnames}
    for rxn_id, rp in rxn_predictions.items():
        loss += 0.5 * rp['weight'] * (rp['e_pred'] - rp['e_ref'])**2
        dloss_tmp = rp['weight'] * (rp['e_pred'] - rp['e_ref'])
        for param in pnames:
            dloss[param] += dloss_tmp * rp['de_pred'][param]
    return loss, dloss

def compute_loss_linear(rxn_predictions, linp):
    nlinp = len(linp)
    b = np.zeros(nlinp)
    cov = np.zeros((nlinp, nlinp))
    cov += 1e-5 * np.identity(nlinp)
    loss = 0
    for rxn_id, rp in rxn_predictions.items():
        diff = rp['e_ref'] - rp['e_pred']
        loss += 0.5 * rp['weight'] * diff**2
        for i in range(nlinp):
            b[i] += rp['weight'] * rp['de_pred'][linp[i]] * diff
            for j in range(nlinp):
                cov[i,j] += rp['weight'] * rp['de_pred'][linp[i]] * rp['de_pred'][linp[j]]
    beta = np.linalg.solve(cov, b)
    beta = {linp[i] : beta[i] for i in range(nlinp)}
    return loss, beta

def train_gd(vpg_func, inputs_dict, formulas, init_params, pweights,
             niter=10, tol=1e-3, rr=1e-3):
    """
    vpg_func: takes (params, inputs, weights) -> Exc, {deriv of Exc wrt params}
    inputs_dict:
    {
        <mol_id> : {
            "xc_inputs" : [rho_a, rho_b, sigma_a, sigma_ab, sigma_b, tau_a, tau_b, exx_a, exx_b],
            "weights" : <array>,
            "e_base" : <float>
        }
    }
    """
    params = copy.deepcopy(init_params)
    params = {k:np.float64(v) for k,v in params.items()}
    pnames = list(params.keys())
    old_params = copy.deepcopy(params)
    converged = False
    param_diffs = {}
    for iter_num in range(niter):
        mol_predictions = compute_mol_preds_and_derivs(vpg_func, params, inputs_dict)
        rxn_predictions = compute_rxn_preds_and_derivs(formulas, mol_predictions, pnames)
        loss, dloss = compute_loss_and_grad(rxn_predictions, pnames)
        print("LOSS AT ITER={}: {}".format(iter_num, loss))
        converged = True
        for param in pnames:
            params[param] -= rr * pweights[param] * dloss[param]
            param_diffs[param] = params[param] - old_params[param]
            if abs(param_diffs[param] * pweights[param]) > tol:
                converged = False
        if converged:
            break
        old_params = copy.deepcopy(params)
    return loss, params, param_diffs, converged

def train_lr_plus_gd(vpg_func, inputs_dict, formulas, init_params, pweights,
                     niter=10, tol=1e-3, rr=1e-3):
    params = copy.deepcopy(init_params)
    params = {k:np.float64(v) for k,v in params.items()}
    pnames = list(params.keys())
    linp = [p for p in pnames if p.startswith('X')]
    nlinp = len(linp)
    old_params = copy.deepcopy(params)
    converged = False
    param_diffs = {}
    for iter_num in range(niter):
        for p in linp:
            params[p] = np.float64(0)
        mol_predictions = compute_mol_preds_and_derivs(vpg_func, params, inputs_dict)
        rxn_predictions = compute_rxn_preds_and_derivs(formulas, mol_predictions, pnames)
        loss, beta = compute_loss_linear(rxn_predictions, linp)
        for p in linp:
            params[p] = beta[p]
        print("LOSS 1 AT ITER={}: {}, PARAMS: {}".format(iter_num, loss, params))
        mol_predictions = compute_mol_preds_and_derivs(vpg_func, params, inputs_dict)
        rxn_predictions = compute_rxn_preds_and_derivs(formulas, mol_predictions, pnames)
        loss, dloss = compute_loss_and_grad(rxn_predictions, pnames)
        print("LOSS 2 AT ITER={}: {}, PARAMS: {}".format(iter_num, loss, params))
        converged = True
        for param in pnames:
            params[param] -= rr * pweights[param] * dloss[param]
            param_diffs[param] = params[param] - old_params[param]
            if abs(param_diffs[param] * pweights[param]) > tol:
                converged = False
        if converged:
            break
        old_params = copy.deepcopy(params)
    return loss, params, param_diffs, converged

def train_bfgs(vpg_func, inputs_dict, formulas, init_params, pweights,
               niter=10, tol=1e-3, rr=1.0, mul_pweights=True,
               train_method='BFGS'):
    pnames = list(init_params.keys())
    pnames.sort()
    def get_loss(params):
        if mul_pweights:
            params = {k:v*pweights[k] for k,v in zip(pnames, params)}
        else:
            params = {k:v for k,v in zip(pnames, params)}
        mol_predictions = compute_mol_preds_and_derivs(vpg_func, params, inputs_dict)
        rxn_predictions = compute_rxn_preds_and_derivs(formulas, mol_predictions, pnames)
        loss, dloss = compute_loss_and_grad(rxn_predictions, pnames)
        if mul_pweights:
            dloss = [dloss[k]*pweights[k] for k in pnames]
        else:
            dloss = [dloss[k] for k in pnames]
        dloss = [dl*rr for dl in dloss]
        print("CURRENT LOSS", loss)
        print(pnames)
        print(params)
        return rr*np.asarray(loss), np.asarray(dloss)
    if mul_pweights:
        init_params = np.array([init_params[k]/pweights[k] for k in pnames])
    else:
        init_params = np.array([init_params[k] for k in pnames])
    from scipy.optimize import minimize
    ## bounds = [(pweights[k]/2,pweights[k]*2) for k in pnames]
    res = minimize(get_loss, init_params, method=train_method, jac=True, options={'gtol': 1e-7, 'maxfun': 300, 'maxiter': 300})
    if mul_pweights:
        params = {k:v*pweights[k] for k,v in zip(pnames, res.x)}
    else:
        params = {k:v for k,v in zip(pnames, res.x)}
    param_diffs = {k:v for k,v in zip(pnames, res.jac)}
    return res.fun / rr, params, param_diffs, res.success

def main():
    m_desc = 'Train a parametric (JAX-implemented) XC functional'

    XC_MODELS = ['PW6B95', 'PW8B95']

    parser = ArgumentParser(description=m_desc)
    xcdesc = 'Name of parametric model for XC, supported: {}'.format(XC_MODELS)
    parser.add_argument('reaction_datasets', type=str, nargs='+')
    parser.add_argument('--xc-model-name', type=str, default='PW6B95',
                        help=xcdesc)
    #parser.add_argument('--x-model-name', type=str, default=None,
    #                    help='Model for exchange, defaults to HF. If CIDER model, Analysis object must contain precomputed features')
    parser.add_argument('--basis', type=str, default='def2-qzvppd',
                        help='Basis set code for which reference data was computed')
    parser.add_argument('--functional', type=str, default='PBE',
                        help='XC functional data for which reference data was computed')
    parser.add_argument('--save-file', type=str, default=None,
                        help='If not None, save parameters to this file as yaml')
    parser.add_argument('--d4-functional', type=str, default=None,
                        help='Functional for parametrizing D4 correction, default no correction')
    parser.add_argument('--relative-train-rate', type=float, default=1.0,
                        help='Relative training rate')
    parser.add_argument('--param-weights-file', type=str, default=None,
                        help='File for weights determining parameter training weights')
    parser.add_argument('--niter', type=int, default=10,
                        help='Maximum number of iterations')
    parser.add_argument('--rtol', type=float, default=1e-3,
                        help='Relative tolerance (gets weighted by pweights) for convergence')
    parser.add_argument('--mlfunc-path', type=str, default=None,
                        help='If not None, replace exact exchange with the prediction of this CIDER model')
    parser.add_argument('--train-method', type=str, default='BFGS')
    parser.add_argument('--init-param-file', type=str, default=None)
    args = parser.parse_args()

    print("TRAIN INPUTS")
    print(args)

    formulas = {}
    for rxn_set in args.reaction_datasets:
        formulas.update(load_rxns(rxn_set))
    """
    rnames = list(formulas.keys())
    np.random.seed(42)
    np.random.shuffle(rnames)
    new_formulas = {}
    for r in rnames[:20]:
        new_formulas[r] = formulas[r]
    formulas = new_formulas
    """

    mol_ids = set()
    for _, rxn in formulas.items():
        for struct in rxn['structs']:
            mol_ids.add(struct)
    mol_ids = list(mol_ids)
    mol_ids.sort()

    mol_data = {}
    for mol_id in mol_ids:
        mol_data[mol_id] = load_molecular_data(
            args.basis, args.functional, mol_id,
            d4_functional=args.d4_functional
        )

    model_type = args.xc_model_name.upper().strip()
    if model_type == 'PW6B95':
        nargs = 9
        xcfunc = pw6b95_train
        init_params = copy.deepcopy(PW6B95_DEFAULT_PARAMS)
    elif model_type == 'PW8B95':
        nargs = 9
        xcfunc = pw8b95
        init_params = copy.deepcopy(PW8B95_DEFAULT_PARAMS)
    elif model_type == 'PW11B95':
        nargs = 9
        xcfunc = pw11b95
        init_params = copy.deepcopy(PW11B95_DEFAULT_PARAMS)
    elif model_type == 'PW12B95':
        nargs = 9
        xcfunc = pw12b95
        init_params = copy.deepcopy(PW12B95_DEFAULT_PARAMS)
    elif model_type == 'RPW12B95':
        nargs = 9
        xcfunc = rpw12b95
        init_params = copy.deepcopy(PW12B95_DEFAULT_PARAMS)
    elif model_type == 'PW13B95':
        nargs = 9
        xcfunc = pw13b95
        init_params = copy.deepcopy(PW13B95_DEFAULT_PARAMS)
    elif model_type == 'PW14B95':
        nargs = 9
        xcfunc = pw14b95
        init_params = copy.deepcopy(PW14B95_DEFAULT_PARAMS)
    else:
        raise ValueError('Unsupported functional model')
    if args.init_param_file is not None:
        with open(args.init_param_file, 'r') as f:
            init_params = yaml.load(f, Loader=yaml.Loader)
    vpg_func = build_xcfunc_and_param_grad(xcfunc, nargs)

    if args.param_weights_file is not None:
        with open(args.param_weights_file, 'r') as f:
            pweights = yaml.load(f, Loader=yaml.Loader)
    else:
        pweights = {p:1.0 for p in list(init_params.keys())}
    NTR = len(formulas)
    for p in list(pweights.keys()):
        pweights[p] /= NTR

    if args.mlfunc_path is not None:
        import joblib
        mlfunc = joblib.load(os.path.expanduser(args.mlfunc_path))
        inputs_dict = get_jax_inputs_dict(mol_data, mlfunc=mlfunc)
    else:
        inputs_dict = get_jax_inputs_dict(mol_data)

    if 'builtin' in args.train_method:
        if args.train_method == 'builtin_gd':
            train = train_gd
        else:
            train = train_lr_plus_gd
        loss, params, params_diff, converged = train(
            vpg_func, inputs_dict, formulas, init_params, pweights,
            niter=args.niter, tol=args.rtol, rr=args.relative_train_rate
        )
    else:
        train = train_bfgs
        loss, params, params_diff, converged = train(
            vpg_func, inputs_dict, formulas, init_params, pweights,
            niter=args.niter, tol=args.rtol, rr=args.relative_train_rate,
            train_method=args.train_method
        )
    result = {
        'converged': converged,
        'loss': loss,
        'params': params,
        'params_diff': params_diff,
    }
    print("RESULT")
    print(result)
    print(yaml.dump(result))
    if args.save_file is not None:
        if not args.save_file.endswith('.yaml'):
            args.save_file = args.save_file + '.yaml'
        with open(args.save_file, 'w') as f:
            yaml.dump(result, f)


if __name__ == '__main__':
    main()
