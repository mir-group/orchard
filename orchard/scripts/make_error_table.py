from mldftdat.data import predict_exchange, predict_correlation,\
                          predict_total_exchange_unrestricted
from mldftdat.analyzers import ElectronAnalyzer
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
    args = parser.parse_args()

    mol_ids = load_mol_ids(args.mol_file)

    Analyzer = ElectronAnalyzer    

    dirs = []
    for mol_id in mol_ids:
        dirs.append(get_save_dir(SAVE_ROOT, 'KS', args.basis,
                                 mol_id, args.functional))

    rows, models = load_models(args.model_file)

    if args.version == '3':
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

