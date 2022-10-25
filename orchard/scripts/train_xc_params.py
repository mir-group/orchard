
from orchard.workflow_utils import load_rxns
from orchard.workflow_utils import SAVE_ROOT
import os, yaml
import dftd4.pyscf as pyd4
import numpy as np

from mldftdat.models.jax_pw6b95 import pw6b95_train, pw8b95, \
    build_xcfunc_and_param_grad, PW6B95_DEFAULT_PARAMS, \
    PW8B95_DEFAULT_PARAMS 


def get_base_energy(analyzer, d4func=None):
    d4func = settings['control'].get('dftd4_functional')
    restricted = True if analyzer.dm.ndim == 2 else False
    if restricted:
        calc = scf.RHF(analyzer.mol)
    else:
        calc = scf.UHF(analyzer.mol)
    dm = analyzer.dm
    h1e = calc.get_hcore()
    e_base = calc.energy_nuc()
    e_base+= numpy.einsum('ij,ji->', h1e, dm).real
    e_base+= np.dot(analyzer.grids.weights, analyzer.get('ha_energy_density'))
    if d4func is not None:
        e_base += pyd4.DFTD4Dispersion(
            calc.mol, xc=d4func.upper().replace(" ", "")
        ).kernel()[0]
    return e_base

def load_molecular_data(basis, functional, mol_id, d4_functional=None):
    print('MOL LOAD', mol_id)
    d = os.path.join('SAVE_ROOT', 'KS', functional, basis, mol_id, 'analysis_L1.hdf5')
    analyzer = Analyzer.load(d)
    analyzer.set('restricted', analyzer.dm.ndim == 2)
    analyzer.set('e_base', get_base_energy(analyzer, d4_functional))
    analyzer.set('grids', analyzer.grids)
    analyzer.set('mol', analyzer.mol)
    return analyzer._data

def get_jax_inputs_dict(mol_data):
    inputs_dict = {}
    _aca = np.ascontiguousarray
    tol = 1e-9
    for mol_id, data in mol_data.items():
        print('MOL SETUP', mol_id)
        inp = {}
        inp['weights'] = data['grids'].weights
        inp['e_base'] = data['e_base']
        rho = data['rho_data']
        exx = data['ex_energy_density']
        if data['restricted']:
            cond = rho[0] > tol
            sigma = 0.25 * np.einsum('xg,xg->g', rho[1:4,cond], rho[1:4,cond])
            inp['xc_inputs'] = [
                _aca(0.5 * rho[0,cond]), _aca(0.5 * rho[0,cond]),
                sigma, sigma, sigma,
                _aca(0.5 * rho[5,cond]), _aca(0.5 * rho[5,cond]),
                _aca(0.5 * exx[cond]), _aca(0.5 * exx[cond]),
            ]
        else:
            cond = rho[:,0].sum(axis=0) > tol
            sigma_aa = np.einsum('xg,xg->g', rho[0,1:4,cond], rho[0,1:4,cond])
            sigma_ab = np.einsum('xg,xg->g', rho[0,1:4,cond], rho[1,1:4,cond])
            sigma_bb = np.einsum('xg,xg->g', rho[1,1:4,cond], rho[1,1:4,cond])
            inp['xc_inputs'] = [
                _aca(rho[0,0,cond]), _aca(rho[1,0,cond]),
                sigma_aa, sigma_ab, sigma_bb,
                _aca(rho[0,5,cond]), _aca(rho[1,5,cond]),
                _aca(exx[0,cond]), _aca(exx[1,cond]),
            ]
        inputs_dict[mol_id] = inp
    return inputs_dict


def compute_mol_preds_and_derivs(vpg_func, params, inputs_dict):
    predictions = {}
    for mol_id, data in inputs_dict.items():
        print('MOL-JAX', mol_id)
        exc, dexc = vpg_func(params, data['xc_inputs'], data["weights"])
        predictions[mol_id] = {"energy": e_base + exc, "grad": dexc}
    return predictions

def compute_rxn_preds_and_derivs(formulas, mol_predictions, pnames):
    ha_per_kcal = 0.001593601
    rxn_predictions = {}
    for rxn_id, rxn in formulas.items():
        print('RXN', rxn_id)
        rp = {
            'e_pred' : 0,
            'de_pred' : {param : 0 for param in pnames},
            'e_ref' : ha_per_kcal * count * formulas[rxn_id]['energy']
            'weight' : 1.0 / (formulas[rxn_id].get('noise_factor') or 1.0)**2
        }
        for struct, count in zip(rxn['structs'], rxn['counts']):
            mp = mol_predictions[struct]
            rp['e_pred'] += count * mp['energy']
            for param in pnames:
                rp['de_pred'][param] += count * mp['grad']
        rxn_predictions[rxn_id] = rp
    return rxn_predictions

def compute_loss_and_grad(rxn_predictions, pnames):
    loss = 0
    dloss = {param:0 for param in pnames}
    for rxn_id, rp in formulas.items():
        loss += 0.5 * rp['weight'] * (rp['e_pred'] - rp['e_ref'])**2
        dloss_tmp = rp['weight'] * (rp['e_pred'] - rp['e_ref'])
        for param in pnames:
            dloss[param] += dloss_tmp * rp['de_pred'][param]
    return loss, dloss

def train(vpg_func, inputs_dict, formulas, init_params, pweights,
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
    pnames = list(params.keys())
    old_params = copy.deepcopy(params)
    converged = False
    param_diffs = {}
    for iter_num in range(niter):
        mol_predictions = compute_mol_preds_and_derivs(vpg_func, params, inputs_dict)
        rxn_predictions = compute_rxn_preds_and_derivs(formulas, mol_predictions, pnames)
        loss, dloss = compute_loss_and_grad(rxn_predictions, pnames)
        print("LOSS AT ITER={}: {}".format(iter_num, loss))
        for param in pnames:
            params[param] -= rr * pweights[param] * dloss[param]
            param_diffs[param] = params[param] - old_params[param]
            if abs(param_diffs[param] * pweights[param]) > tol:
                break
        else:
            converged = True
            break
        old_params = copy.deepcopy(params)
    return loss, params, param_diffs, converged

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
    parser.add_argument('--relative-train-rate', type=float, default=1e-3,
                        help='Relative training rate')
    parser.add_argument('--param-weights-file', type=str, default=None,
                        help='File for weights determining parameter training weights')
    parser.add_argument('--niter', type=int, default=10,
                        help='Maximum number of iterations')
    parser.add_argument('--rtol', type=float, default=1e-3,
                        help='Relative tolerance (gets weighted by pweights) for convergence')
    parser.add_argument()
    args = parser.parse_args()

    Analyzer = ElectronAnalyzer

    for rxn_set in args.reaction_datasets:
        formulas.update(load_rxns(rxn_set))

    mol_ids = set()
    for rxn in formulas:
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
    else:
        raise ValueError('Unsupported functional model')
    vpg_func = build_xcfunc_and_param_grad(xcfunc, nargs)

    if args.param_weights_file is not None:
        with open(param_weights_file, 'r') as f:
            pweights = yaml.load(f, Loader=yaml.Loader)
    else:
        pweights = {p:1.0 for p in list(init_params.keys())}
    NTR = len(formulas)
    for p in list(pweights.keys()):
        pweights[p] /= NTR

    inputs_dict = get_jax_inputs_dict(mol_data)

    loss, params, param_diffs, converged = train(
        vpg_func, inputs_dict, formulas, init_params, pweights,
        niter=args.niter, tol=args.rtol, rr=args.relative_train_rate
    )
    result = {
        'converged': converged,
        'loss': loss,
        'params': params,
        'params_diff': params_diff,
    }
    print("RESULT")
    print(result)
    if args.save_file is not None:
        if not args.save_file.endswith('.yaml'):
            args.save_file = args.save_file + '.yaml'
        with open(args.save_file, 'w') as f:
            yaml.dump(result, f)


if __name__ == '__main__':
    main()
