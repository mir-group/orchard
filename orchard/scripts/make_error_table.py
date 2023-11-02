from ciderpress.data import predict_exchange, predict_total_exchange_unrestricted
from ciderpress.analyzers import ElectronAnalyzer
from ciderpress.models.compute_mol_cov import compute_x_pred
from orchard.workflow_utils import get_save_dir, SAVE_ROOT, load_mol_ids
import numpy as np 
from collections import Counter
from ase.data import chemical_symbols, atomic_numbers,\
                     ground_state_magnetic_moments
from argparse import ArgumentParser
import pandas as pd
from joblib import dump, load
import yaml
import os
import sys

def load_models(model_file):
    with open(model_file, 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
        names = []
        models = []
        for name in d:
            names.append(name)
            if d[name] is None:
                models.append(name)
            elif os.path.isfile(d[name]):
                models.append(load(d[name]))
                print(models[-1].desc_version, models[-1].amin, models[-1].a0, models[-1].fac_mul)
            elif os.path.isfile(os.path.join(os.environ.get('DATADIR'), d[name])):
                models.append(load(os.path.join(os.environ.get('DATADIR'), d[name])))
            else:
                models.append(d[name])
        return names, models

def error_table3(dirs, Analyzer, models, rows, basis, functional):
    errlst = [[] for _ in models]
    ae_errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    ae_fxlst_pred = [[] for _ in models]
    fxlst_true = []
    ae_fxlst_true = []
    count = 0
    NMODEL = len(models)
    ise = np.zeros(NMODEL)
    tse = np.zeros(NMODEL)
    rise = np.zeros(NMODEL)
    rtse = np.zeros(NMODEL)
    for d in dirs:
        print(d.split('/')[-1])
        analyzer = Analyzer.load(os.path.join(d, 'analysis_L3.hdf5'))
        atoms = [atomic_numbers[a[0]] for a in analyzer.mol._atom]
        formula = Counter(atoms)
        element_analyzers = {}
        for Z in list(formula.keys()):
            symbol = chemical_symbols[Z]
            spin = int(ground_state_magnetic_moments[Z])
            path = '{}/KS/{}/{}/atoms/{}-{}-{}/analysis_L3.hdf5'.format(
                        SAVE_ROOT, functional, basis, Z, symbol, spin)
            element_analyzers[Z] = ElectronAnalyzer.load(path)
        weights = analyzer.grids.weights
        rho = analyzer.rho_data[0,:]
        assert analyzer.grids.level == 3
        condition = rho > 3e-5
        fx_total_ref_true = 0
        for Z in list(formula.keys()):
            fx_total_ref_true += formula[Z] \
                                 * predict_total_exchange_unrestricted(
                                        element_analyzers[Z])
        xef_true, eps_true, neps_true, fx_total_true = \
            predict_exchange(analyzer)
        fxlst_true.append(fx_total_true)
        ae_fxlst_true.append(fx_total_true - fx_total_ref_true)
        count += eps_true.shape[0]
        for i, model in enumerate(models):
            fx_total_ref = 0
            for Z in list(formula.keys()):
                fx_total_ref += formula[Z] \
                                * predict_total_exchange_unrestricted(
                                    element_analyzers[Z],
                                    model=model)
            xef_pred, eps_pred, neps_pred, fx_total_pred = \
                predict_exchange(analyzer, model=model)
            print(fx_total_pred, fx_total_true,
                fx_total_ref, fx_total_ref_true)
            print(fx_total_pred - fx_total_true,
                  fx_total_pred - fx_total_true \
                  - (fx_total_ref - fx_total_ref_true))

            ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2,
                             weights[condition])
            tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()
            rise[i] += np.dot((xef_pred[condition] - xef_true[condition])**2,
                              weights[condition])
            rtse[i] += ((xef_pred[condition] - xef_true[condition])**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            ae_fxlst_pred[i].append(fx_total_pred - fx_total_ref)
            errlst[i].append(fx_total_pred - fx_total_true)
            ae_errlst[i].append(fx_total_pred - fx_total_true \
                                - (fx_total_ref - fx_total_ref_true))
        print()
    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)
    ae_errlst = np.array(ae_errlst)

    print(count, len(dirs))

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    ae_fx_total_rmse = np.sqrt(np.mean(ae_errlst**2, axis=1))
    rmise = np.sqrt(ise / len(dirs))
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / count)

    columns = ['RMSE AEX', 'RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    errtbl = np.array([ae_fx_total_rmse, fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst, ae_errlst),\
           (columns, rows, errtbl)

def error_table3u(dirs, Analyzer, models, rows, basis, functional):
    errlst = [[] for _ in models]
    ae_errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    ae_fxlst_pred = [[] for _ in models]
    fxlst_true = []
    ae_fxlst_true = []
    count = 0
    NMODEL = len(models)
    ise = np.zeros(NMODEL)
    tse = np.zeros(NMODEL)
    rise = np.zeros(NMODEL)
    rtse = np.zeros(NMODEL)
    for d in dirs:
        print(d.split('/')[-1])
        analyzer = Analyzer.load(os.path.join(d, 'analysis_L3.hdf5'))
        atoms = [atomic_numbers[a[0]] for a in analyzer.mol._atom]
        formula = Counter(atoms)
        element_analyzers = {}
        for Z in list(formula.keys()):
            symbol = chemical_symbols[Z]
            spin = int(ground_state_magnetic_moments[Z])
            path = '{}/KS/{}/{}/atoms/{}-{}-{}/analysis_L3.hdf5'.format(
                        SAVE_ROOT, functional, basis, Z, symbol, spin)
            element_analyzers[Z] = ElectronAnalyzer.load(path)
        weights = analyzer.grids.weights
        rho = analyzer.rho_data[0,:]
        condition = rho > 3e-5
        fx_total_ref_true = 0
        for Z in list(formula.keys()):
            fx_total_ref_true += formula[Z] \
                                 * predict_total_exchange_unrestricted(
                                        element_analyzers[Z])
        fx_total_true = \
            predict_total_exchange_unrestricted(analyzer)
        fxlst_true.append(fx_total_true)
        ae_fxlst_true.append(fx_total_true - fx_total_ref_true)
        count += 1
        for i, model in enumerate(models):
            fx_total_ref = 0
            for Z in list(formula.keys()):
                fx_total_ref += formula[Z] \
                                * predict_total_exchange_unrestricted(
                                    element_analyzers[Z],
                                    model=model)
            fx_total_pred = \
                predict_total_exchange_unrestricted(analyzer, model=model)
            print(fx_total_pred, fx_total_true,
                fx_total_ref, fx_total_ref_true)
            print(fx_total_pred - fx_total_true,
                  fx_total_pred - fx_total_true \
                  - (fx_total_ref - fx_total_ref_true))

            fxlst_pred[i].append(fx_total_pred)
            ae_fxlst_pred[i].append(fx_total_pred - fx_total_ref)
            errlst[i].append(fx_total_pred - fx_total_true)
            ae_errlst[i].append(fx_total_pred - fx_total_true \
                                - (fx_total_ref - fx_total_ref_true))
        print()
    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)
    ae_errlst = np.array(ae_errlst)

    print(count, len(dirs))

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    ae_fx_total_rmse = np.sqrt(np.mean(ae_errlst**2, axis=1))
    rmise = np.sqrt(ise / len(dirs))
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / count)

    columns = ['RMSE AEX', 'RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    errtbl = np.array([ae_fx_total_rmse, fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst, ae_errlst),\
           (columns, rows, errtbl)

def get_single_file_xpred(fname_base, models, _get_fname):
    tmp = '//SUFFIX_TEMPLATE//'
    default = 'WIDE_WIDE'
    try:
        fname = _get_fname(fname_base.replace(tmp, default))
        exx = compute_x_pred(fname, 'EXX')
    except:
        default = 'WIDE'
        fname = _get_fname(fname_base.replace(tmp, default))
        exx = compute_x_pred(fname, 'EXX')
    diffs = []
    for model in models:
        if isinstance(model, str):
            fname = _get_fname(fname_base.replace(tmp, default))
            xpred = compute_x_pred(fname, 'SL', model=model)
        else:
            fname = _get_fname(fname_base.replace(tmp, model.args.suffix))
            xpred = compute_x_pred(fname, 'ML', model=model)
        diffs.append(xpred - exx)
    return exx, diffs

def error_table_rxn(mol_ids, fname_base, models, formulas=None, extra_dirs=None):
    def _get_fname(fname):
        if args.extra_dirs is None:
            fname = os.path.join(SAVE_ROOT, 'DATASETS', args.functional,
                                 args.basis, args.desc_version, fname)
        else:
            ddirs = [SAVE_ROOT] + args.extra_dirs
            for dd in ddirs:
                cdd = os.path.join(dd, 'DATASETS', args.functional,
                                  args.basis, args.desc_version, fname)
                print(cdd)
                if os.path.exists(cdd):
                    fname = cdd
                    break
            else:
                raise FileNotFoundError('Could not find dataset in provided dirs: {}'.format(fname))
        return fname
    
    """
    tmp = '//SUFFIX_TEMPLATE//'
    default = 'WIDE_WIDE'
    fname = _get_fname(fname_base.replace(tmp, default))
    exx = compute_x_pred(fname, 'EXX')
    diffs = []
    for model in models:
        if isinstance(model, str):
            fname = _get_fname(fname_base.replace(tmp, default))
            xpred = compute_x_pred(fname, 'SL', model=model)
        else:
            fname = _get_fname(fname_base.replace(tmp, model.args.suffix))
            xpred = compute_x_pred(fname, 'ML', model=model)
        diffs.append(xpred - exx)
    if formulas is None:
        return exx, np.array(diffs)
    """
    exx, diffs = None, None
    for fb in fname_base:
        exx_tmp, diffs_tmp = get_single_file_xpred(fb, models, _get_fname)
        if exx is None:
            exx = exx_tmp
            diffs = diffs_tmp
        else:
            exx = np.append(exx, exx_tmp)
            diffs = np.append(diffs, diffs_tmp, axis=1)
    if formulas is None:
        return exx, np.array(diffs)

    rxn_names = list(formulas.keys())
    rxn_names.sort()
    rxn_diffs = []
    noise_factors = [formulas[n].get('noise_factor') or 1.0 for n in rxn_names]
    for im, model in enumerate(models):
        ndiffs = {m:x for m,x in zip(mol_ids, diffs[im])}
        rxn_diff_list = []
        for name in rxn_names:
            rxn_diff = 0
            for s, c in zip(formulas[name]['structs'], formulas[name]['counts']):
                rxn_diff += c * ndiffs[s]
            rxn_diff_list.append(rxn_diff)
        rxn_diffs.append(np.array(rxn_diff_list))
    return exx, diffs, rxn_names, np.array(rxn_diffs), np.array(noise_factors, dtype=np.float64)

def error_table_corr(dirs, Analyzer, models, rows):
    from collections import Counter
    from ase.data import chemical_symbols, atomic_numbers, ground_state_magnetic_moments
    errlst = [[] for _ in models]
    ae_errlst = [[] for _ in models]
    fxlst_pred = [[] for _ in models]
    ae_fxlst_pred = [[] for _ in models]
    fxlst_true = []
    ae_fxlst_true = []
    count = 0
    NMODEL = len(models)
    ise = np.zeros(NMODEL)
    tse = np.zeros(NMODEL)
    rise = np.zeros(NMODEL)
    rtse = np.zeros(NMODEL)
    for d in dirs:
        print(d.split('/')[-1])
        analyzer = Analyzer.load(os.path.join(d, 'data.hdf5'))
        if 'rho_data' not in analyzer.keys():
            analyzer.get_rho_data()
        atoms = [atomic_numbers[a[0]] for a in analyzer.mol._atom]
        formula = Counter(atoms)
        element_analyzers = {}
        for Z in list(formula.keys()):
            symbol = chemical_symbols[Z]
            spin = int(ground_state_magnetic_moments[Z])
            letter = '' if spin == 0 else 'U'
            path = '{}/{}CCSD/aug-cc-pvtz/atoms/{}-{}-{}/data.hdf5'.format(
                        SAVE_ROOT, letter, Z, symbol, spin)
            if letter == '':
                element_analyzers[Z] = CCSDAnalyzer.load(path)
            else:
                element_analyzers[Z] = UCCSDAnalyzer.load(path)
            if 'rho_data' not in element_analyzers[Z].keys():
                element_analyzers[Z].get_rho_data()
        weights = analyzer.grids.weights
        rho = analyzer.rho_data[0,:]
        condition = rho > 3e-5
        fx_total_ref_true = 0
        for Z in list(formula.keys()):
            restricted = True if type(element_analyzers[Z]) == CCSDAnalyzer else False
            _, _, fx_total_ref_tmp = predict_correlation(
                                        element_analyzers[Z],
                                        restricted=restricted)
            fx_total_ref_true += formula[Z] * fx_total_ref_tmp
        eps_true, neps_true, fx_total_true = \
            predict_correlation(analyzer)
        fxlst_true.append(fx_total_true)
        ae_fxlst_true.append(fx_total_true - fx_total_ref_true)
        count += eps_true.shape[0]
        for i, model in enumerate(models):
            fx_total_ref = 0
            for Z in list(formula.keys()):
                restricted = True if type(element_analyzers[Z]) == CCSDAnalyzer else False
                _, _, fx_total_tmp = predict_correlation(
                                        element_analyzers[Z],
                                        model=model, restricted=restricted)
                fx_total_ref += formula[Z] * fx_total_tmp
            eps_pred, neps_pred, fx_total_pred = \
                predict_correlation(analyzer, model=model)
            print(fx_total_pred - fx_total_true,
                  fx_total_pred - fx_total_true \
                  - (fx_total_ref - fx_total_ref_true))

            ise[i] += np.dot((eps_pred[condition] - eps_true[condition])**2,
                             weights[condition])
            tse[i] += ((eps_pred[condition] - eps_true[condition])**2).sum()

            fxlst_pred[i].append(fx_total_pred)
            ae_fxlst_pred[i].append(fx_total_pred - fx_total_ref)
            errlst[i].append(fx_total_pred - fx_total_true)
            ae_errlst[i].append(fx_total_pred - fx_total_true \
                                - (fx_total_ref - fx_total_ref_true))
        print(errlst[-1][-1], ae_errlst[-1][-1])
        print()
    fxlst_true = np.array(fxlst_true)
    fxlst_pred = np.array(fxlst_pred)
    errlst = np.array(errlst)
    ae_errlst = np.array(ae_errlst)

    print(count, len(dirs))

    fx_total_rmse = np.sqrt(np.mean(errlst**2, axis=1))
    ae_fx_total_rmse = np.sqrt(np.mean(ae_errlst**2, axis=1))
    rmise = np.sqrt(ise / len(dirs))
    rmse = np.sqrt(tse / count)
    rrmise = np.sqrt(rise / len(dirs))
    rrmse = np.sqrt(rtse / count)

    columns = ['RMSE AEX', 'RMSE EX', 'RMISE', 'RMSE', 'Rel. RMISE', 'Rel. RMSE']
    errtbl = np.array([ae_fx_total_rmse, fx_total_rmse, rmise, rmse, rrmise, rrmse]).transpose()

    return (fxlst_true, fxlst_pred, errlst, ae_errlst),\
           (columns, rows, errtbl)


if __name__ == '__main__':
    m_desc = 'Compute, print, and return errors of different methods for prediction of exchange and correlation energies.'

    parser = ArgumentParser(description=m_desc)
    parser.add_argument('version', type=str,
                        help=('1, 2, 3, u, or c.\n'
                        '1: Total exchange error for spin-restricted systems\n'
                        '2: Same as above but also returns data for ML descriptors\n'
                        '3: Total and atomization exchange error for spin-restricted systems\n'
                        'u: Total exchange error for spin-unrestricted systems\n'
                        'c: Total correlation exchange error.'))
    parser.add_argument('model_file', type=str,
                        help='yaml file containing list of models and how to load them.')
    parser.add_argument('mol_file', type=str,
                        help='yaml file containing list of directories and calc type')
    parser.add_argument('basis', metavar='basis', type=str,
                        help='basis set code')
    parser.add_argument('--functional', metavar='functional', type=str, default=None,
                        help='exchange-correlation functional, HF for Hartree-Fock')
    parser.add_argument('--save-file', type=str, default=None,
                        help='If not None, save error table to this file.')
    parser.add_argument('--xsuffix', type=str, default=None,
                        help='If provided, use stored feat data for avoid recomputing descriptors')
    parser.add_argument('--reaction-dataset', type=str, default=None,
                        help='If supplied, compute chemical reaction diffs')
    parser.add_argument('--desc-version', type=str, default=None)
    parser.add_argument('--extra-dirs', type=str, nargs='+', default=None)
    parser.add_argument('--base-sysdir', type=str, default=None,
                        help='If provided, append to mol_files')
    args = parser.parse_args()

    if args.xsuffix is not None:
        args.xsuffix = '//SUFFIX_TEMPLATE//'
    if args.base_sysdir is None:
        mol_ids = load_mol_ids(args.mol_file)
        fname = os.path.basename(args.mol_file)[:-5] + '_' + args.xsuffix
        fname = fname.upper()
        fnames = [fname]
    else:
        mol_ids = []
        fnames = []
        for f in args.mol_file.split(','):
            mol_ids += load_mol_ids(os.path.join(args.base_sysdir, f))
            fname = os.path.basename(f) + '_' + args.xsuffix
            fname = fname.upper()
            fnames.append(fname)

    Analyzer = ElectronAnalyzer 

    dirs = []
    for mol_id in mol_ids:
        dirs.append(get_save_dir(SAVE_ROOT, 'KS', args.basis,
                                 mol_id, args.functional))

    rows, models = load_models(args.model_file)

    if args.xsuffix is not None:
        formulas = None
        if args.reaction_dataset is not None:
            from orchard.workflow_utils import load_rxns
            formulas = load_rxns(args.reaction_dataset)
        res = error_table_rxn(mol_ids, fnames, models, formulas=formulas)
        df = pd.DataFrame()
        if formulas is None:
            exx, diffs = res
            dat_names = mol_ids
            errs = diffs
        else:
            exx, pred, rxn_names, rxn_diffs, nfs = res
            dat_names = rxn_names
            errs = rxn_diffs
        for i, r in enumerate(rows):
            df[r] = errs[i]
        df.index = dat_names
        df.loc['MAE'] = df.loc[dat_names].abs().mean()
        df.loc['RMSE'] = df.loc[dat_names].pow(2).mean().pow(0.5)
        if formulas is not None:
            df.loc['LOSS'] = (df.loc[dat_names] / nfs[:,None]).pow(2).sum()
        print(df.to_latex())
    elif args.version == '3':
        res1, res2 = error_table3(dirs, Analyzer, models, rows, args.basis, args.functional)
        fxlst_true, fxlst_pred, errlst, ae_errlst = res1
        columns, rows, errtbl = res2
        print(res1)
        for sublst in res1[2]:
            print(np.mean(sublst), np.std(sublst))
        df = pd.DataFrame(errtbl, index=rows, columns=columns)
        print(df.to_latex())
    elif args.version == 'u':
        res1, res2 = error_table3u(dirs, Analyzer, models, rows, args.basis, args.functional)
        fxlst_true, fxlst_pred, errlst, ae_errlst = res1
        columns, rows, errtbl = res2
        print(res1)
        for sublst in res1[2]:
            print(np.mean(sublst), np.std(sublst))
        df = pd.DataFrame(errtbl, index=rows, columns=columns)
        print(df.to_latex())
    
    if args.save_file is not None:
        df.to_csv(args.save_file)

