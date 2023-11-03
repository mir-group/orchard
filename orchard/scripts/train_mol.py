from argparse import ArgumentParser
import os
import numpy as np
from joblib import load, dump
from orchard.workflow_utils import SAVE_ROOT, load_rxns
from ciderpress.models.train import DescParams, MOLGP, strk_to_tuplek
from ciderpress.models.dft_kernel import DFTKernel
from ciderpress.models.baselines import BASELINE_CODES
from ciderpress.xcutil.transform_data import FeatureList
from ciderpress.density import LDA_FACTOR
from pyscf.lib import chkfile
import importlib
import yaml

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


def find_dataset(fname, args):
    fname = '{}_settings.yaml'.format(fname)
    if args.save_dir is None:
        reldir = os.path.join(
            'DATASETS', args.functional, args.basis,
            args.version, args.suffix, fname
        )
        if args.extra_dirs is None:
            fname = os.path.join(SAVE_ROOT, reldir)
        else:
            ddirs = [os.path.join(SAVE_ROOT, 'DATASETS')] + args.extra_dirs
            for dd in ddirs:
                cdd = os.path.join(dd, reldir)
                print(cdd)
                if os.path.exists(cdd):
                    fname = cdd
                    break
            else:
                raise FileNotFoundError('Could not find dataset in provided dirs')
    else:
        absdir = os.path.join(args.save_dir, args.suffix, fname)
        if args.extra_dirs is None:
            fname = absdir
        else:
            ddirs = [os.path.dirname(absdir)] + args.extra_dirs
            for dd in ddirs:
                cdd = os.path.join(dd, fname)
                print(cdd)
                if os.path.exists(cdd):
                    fname = cdd
                    break
            else:
                raise FileNotFoundError('Could not find dataset in provided dirs')
    return os.path.dirname(fname)


def get_plan_module(plan_file):
    if plan_file.startswith('@'):
        plan_module = importlib.import_module(plan_file[1:])
    else:
        assert os.path.exists(plan_file)
        spec = importlib.util.spec_from_file_location('plan_module', plan_file)
        plan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan_module)
    return plan_module


def parse_settings(args):
    fname = args.datasets_list[0]
    if args.save_dir is None:
        dname = os.path.join(SAVE_ROOT, 'DATASETS', args.functional,
                             args.basis, args.version, args.suffix)
    else:
        dname = os.path.join(args.save_dir, args.suffix)
    print(fname)
    with open(os.path.join(dname,
              '{}_settings.yaml'.format(fname)), 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
    args.gg_a0 = d.get('a0')
    args.gg_amin = d.get('amin')
    args.gg_facmul = d.get('fac_mul')
    args.gg_vvmul = d.get('vvmul')


def parse_dataset_for_ctrl(args, i):
    fname = args.datasets_list[2*i]
    n = int(args.datasets_list[2*i+1])
    dirname = find_dataset(fname, args)
    with open(os.path.join(dirname, '{}_settings.yaml'.format(fname)), 'r') as f:
        settings = yaml.load(f, Loader=yaml.CLoader)
        mol_ids = settings['MOL_IDS']
    Xlist = []
    GXOlist = []
    GXUlist = []
    ylist = []
    for mol_id in mol_ids:
        fname = os.path.join(dirname, mol_id + '.hdf5')
        data = chkfile.load(fname, 'train_data')
        cond = data['desc'][:, 0, :] > args.density_cutoff
        print(data['desc'].shape, data['val'].shape)
        y = data['val'][cond] / (LDA_FACTOR * data['desc'][:, 0][cond]**(4.0 / 3)) - 1
        cond = np.all(cond, axis=0)
        X = data['desc'][:, :, cond]
        if 'ddesc' in data:
            ddesc = strk_to_tuplek(data['ddesc'])
            print(ddesc.keys())
            has_ddesc = True
            GXO = ddesc[('O', 0)][1][:, cond]
            GXU = ddesc[('U', 0)][1][:, cond]
        else:
            has_ddesc = False
        exlda = LDA_FACTOR * data['desc'][:, 0, cond]**(4.0 / 3)
        if args.randomize:
            inds = np.arange(X.shape[-1])
            np.random.shuffle(inds)
            X = X[..., inds]
            if has_ddesc:
                GXO = GXO[..., inds]
                GXU = GXU[..., inds]
        Xlist.append(X[..., ::n])
        if has_ddesc:
            GXOlist.append((ddesc[('O', 0)][0], GXO[..., ::n]))
            GXUlist.append((ddesc[('U', 0)][0], GXU[..., ::n]))
        ylist.append(y[::n])
    return Xlist, GXOlist, GXUlist, ylist, args.datasets_list[2*i], mol_ids


def get_fd_x1(kernel, Xlist, DXlist, delta=1e-5):
    if len(Xlist) == 0:
        return 0
    nfeat = Xlist[0].shape[1]
    print('NFEAT', nfeat)
    deriv = 0
    for i in range(nfeat):
        slist = [DX[0] for DX in DXlist]
        IDXlist = [DX[1][i] for DX in DXlist]
        for s, X in zip(slist, Xlist):
            X[s, i, :] += 0.5 * delta
        utmp = kernel.X0Tlist_to_X1array_mul(Xlist, IDXlist)
        for s, X in zip(slist, Xlist):
            X[s, i, :] -= delta
        ltmp = kernel.X0Tlist_to_X1array_mul(Xlist, IDXlist)
        for s, X in zip(slist, Xlist):
            X[s, i, :] += 0.5 * delta
        print(i, utmp[0], ltmp[0])
        deriv += (utmp - ltmp) / delta
    return deriv


def analyze_cov(X1, avg_and_std=None):
    if avg_and_std is None:
        avg = np.mean(X1, axis=0)
        std = np.std(X1, axis=0)
    else:
        avg, std = avg_and_std
    XW = X1 - avg
    XW /= std
    cov = XW.T.dot(XW) / XW.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    #return avg, std, evals, evecs
    print('COV')
    print(avg)
    print(std)
    print(cov)
    print(evals)
    print(evecs)
    return avg, std, cov, evals, evecs


def parse_list(lststr, T=int):
    return [T(substr) for substr in lststr.split(',')]


def main():
    parser = ArgumentParser(description='Trains a GP exchange model to '
                                        'molecular energy difference and '
                                        'orbital energies.')

    parser.add_argument('save_file', type=str,
                        help='file to which to save new GP')
    parser.add_argument('basis', metavar='basis', type=str,
                        help='basis set code')
    parser.add_argument('--kernel-plan-file', type=str,
                        help='Settings file for list of kernels. See '
                        'ciderpress.models.kernel_plans.settings_example.yaml '
                        'for documentation and format.')
    parser.add_argument('--datasets-list', nargs='+',
                        help='Pairs of dataset names and inverse sampling '
                             'densities')
    parser.add_argument('--load-orbs-list', nargs='+',
                        help='Pairs of dataset names and 0 (no) or 1 (yes) for '
                             'whether to load orbital occupation gradients. '
                             'Default is 1 if orbs are in the dataset, else 0. '
                             'This setting overrides defaults.')
    parser.add_argument('--reactions-list', nargs='+',
                        help='Pairs of dataset names and modes for reactions '
                             'files; mode 0=x-only, 1=c-only, 2=xc')
    parser.add_argument('--extra-dirs', nargs='+', default=None,
                        help='Extra dirs to search for datasets if not found '
                             'in SAVE_ROOT')
    parser.add_argument('--functional', metavar='functional', type=str,
                        default=None,
                        help='exchange-correlation functional for reference '
                             'data, HF for Hartree-Fock')
    parser.add_argument('-c', '--density-cutoff', type=float, default=1e-6)
    parser.add_argument('-s', '--seed', help='random seed', default=0, type=int)
    parser.add_argument('-d', '--delete-k', action='store_true',
                        help='Delete L (LL^T=K the kernel matrix) to save disk '
                             'space. Need to refit when reloading to calculate '
                             'covariance.')
    parser.add_argument('-v', '--version', default='b', type=str,
                        help='version of descriptor set. Must be b, d, or e')
    parser.add_argument('--suffix', default=None, type=str,
                        help='customize data directories with this suffix')
    parser.add_argument('--nmax-sparse', type=int, default=None,
                        help='If set, not more than this many points used in '
                             'sparse set.')
    parser.add_argument('--control-tol', type=float, default=-1e-5,
                        help='Reduce control point size for given tol. '
                             'Negative value means to ignore.')
    parser.add_argument('--mol-sigma', type=float, default=np.sqrt(1e-5),
                        help='Standard deviation noise parameter for total '
                             'molecular energy data.')
    parser.add_argument('--mol-heg', action='store_true',
                        help='Include HEG constraint in molecules dataset.')
    parser.add_argument('--scale-override', type=float, default=None)
    parser.add_argument('--scale-mul', type=float, default=1.0)
    parser.add_argument('--length-scale-mul', type=float, nargs='+',
                        default=[1.0],
                        help='Used for automatic length-scale initial guess')
    parser.add_argument('--min-lscale', type=float, default=None,
                        help='Minimum length-scale for GP kernel')
    parser.add_argument('--libxc-baseline', type=str, default=None,
                        help='Baseline libxc functional for the full model')
    parser.add_argument('--mapped-fname', type=str, default=None,
                        help='If not None, map model and same to this file.')
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--debug-model', type=str, default=None,
                        help='Load joblib and print debug')
    parser.add_argument('--debug-spline', type=str, default=None,
                        help='Load joblib and print debug')
    parser.add_argument('--save-dir', default=None, type=str,
                        help='override default save directory for features')
    args = parser.parse_args()
    parse_settings(args)
    if args.debug_model is not None:
        args.debug_model = load(args.debug_model)
    if args.debug_spline is not None:
        args.debug_spline = load(args.debug_spline)

    assert len(args.datasets_list) % 2 == 0
    assert len(args.load_orbs_list) % 2 == 0
    assert len(args.reactions_list) % 2 == 0

    np.random.seed(args.seed)
    datasets_list = args.datasets_list[::2]
    load_orbs_dict = {k : None for k in datasets_list}
    for i, dset in enumerate(args.load_orbs_list[::2]):
        if dset not in load_orbs_dict.keys():
            raise ValueError
        load_orbs_dict[dset] = int(args.load_orbs_list[2 * i + 1])
    nd = len(datasets_list)
    assert nd > 0

    desc_params = DescParams(
        args.version,
        args.gg_a0,
        args.gg_facmul,
        args.gg_amin,
        args.gg_vvmul,
    )
    with open(args.kernel_plan_file, 'r') as f:
        kernel_plans = yaml.load(f, Loader=yaml.CLoader)

    # Construct initial control points set
    Xlist = []
    GXRlist = []
    GXOlist = []
    GXUlist = []
    ylist = []
    molid_map = {}
    for i in range(nd):
        Xlist_tmp, GXOlist_tmp, GXUlist_tmp, y_tmp, dset_name, dset_ids = \
            parse_dataset_for_ctrl(args, i)
        Xlist += Xlist_tmp
        print('LENS', len(GXOlist_tmp), len(Xlist_tmp))
        if len(GXOlist_tmp) == len(Xlist_tmp):
            for XX, XXO, XXU in zip(Xlist_tmp, GXOlist_tmp, GXUlist_tmp):
                assert XX.shape[-1] == XXO[1].shape[-1]
                assert XX.shape[-1] == XXU[1].shape[-1]
            GXRlist += Xlist_tmp
            GXOlist += GXOlist_tmp
            GXUlist += GXUlist_tmp
        ylist += y_tmp
        molid_map[dset_name] = dset_ids
    yctrl = np.concatenate(ylist, axis=0)

    kernels = []
    args.plan_files = []
    mapping_plans = []
    for plan in kernel_plans:
        plan_file = plan.pop('plan_file')
        plan_module = get_plan_module(plan_file)
        args.plan_files.append(plan_file)
        feature_list = FeatureList.load(plan['feature_list'])
        ctrl_tol = plan.get('ctrl_tol') or 1e-5
        ctrl_nmax = plan.get('ctrl_nmax')
        kernels.append(DFTKernel(
            None,
            desc_params,
            feature_list,
            plan['mode'],
            BASELINE_CODES[plan['multiplicative_baseline']],
            additive_baseline=BASELINE_CODES.get(plan['additive_baseline']),
            ctrl_tol=ctrl_tol,
            ctrl_nmax=ctrl_nmax,
            component=plan.get('component'),
        ))
        if 'lscale_override' in plan:
            lscale = np.array(plan.pop('lscale_override'))
            val_pca = None
            deriv_pca = None
        else:
            X1 = kernels[-1].X0Tlist_to_X1array(Xlist)
            DXO1 = get_fd_x1(kernels[-1], GXRlist, GXOlist)
            DXU1 = get_fd_x1(kernels[-1], GXRlist, GXUlist)
            val_pca = analyze_cov(X1)
            analyze_cov(DXO1, avg_and_std=val_pca[:2])
            analyze_cov(DXU1, avg_and_std=val_pca[:2])
            deriv_pca = analyze_cov(DXU1 - DXO1, avg_and_std=val_pca[:2])
            lscale = np.std(X1, axis=0)
            print('SHAPES', X1.shape, yctrl.shape)
        kernel = plan_module.get_kernel(
            natural_scale=np.var(yctrl) if args.scale_override is None else args.scale_override,
            natural_lscale=lscale,
            scale_factor=args.scale_mul,
            lscale_factor=args.length_scale_mul,
            val_pca=val_pca,
            deriv_pca=deriv_pca,
        )
        kernels[-1].set_kernel(kernel)
        if 'mapping_plan' in dir(plan_module):
            mfunc = plan_module.mapping_plan
        else:
            mfunc = None
        mapping_plans.append(mfunc)

    gpr = MOLGP(
        kernels,
        desc_params,
        libxc_baseline=args.libxc_baseline,
        default_noise=args.mol_sigma,
    )
    gpr.args = args

    # Set the control points in the model
    gpr.set_control_points(Xlist, reduce=True)
    print('CTRL SIZE', kernels[-1].X1ctrl.shape)

    rxn_list = []
    rxn_id_list = []
    for i, rxn_id in enumerate(args.reactions_list[::2]):
        mode = int(args.reactions_list[2 * i + 1])
        rxn_dict = load_rxns(rxn_id)
        for k, v in rxn_dict.items():
            rxn_id_list.append(k)
            rxn_list.append((mode, v))

    for i in range(nd):
        fname = datasets_list[i]
        load_orbs = load_orbs_dict[fname]
        mol_ids = molid_map[fname]
        fname = find_dataset(fname, args)
        gpr.store_mol_covs(fname, mol_ids, get_orb_deriv=load_orbs,
                           get_correlation=True)
    # TODO add utilities for saving exact constraints!
    #if args.mol_heg:
    #    vw_tmp, exx_tmp = compute_heg_covs(model)
    #    vwrtt_list.append(vw_tmp)
    #    exx_list.append(exx_tmp)
    #    system_ids.append('UNIFORM_ELECTRON_GAS')
    #    rxn_list.append((
    #        0,
    #        {'structs': ['UNIFORM_ELECTRON_GAS'],
    #         'counts': [1], 'noise': 0.00}
    #    ))

    gpr.add_reactions(rxn_list)

    gpr.fit()

    K = gpr.Kcov_
    y = gpr.y_mol_
    alpha = gpr.alpha_mol_
    y_pred = K.dot(alpha)
    dy = y_pred - y
    rtkd = np.sqrt(np.diag(K))
    rxn_id_arr = np.array(rxn_id_list)
    nitems = 20
    for i, rxn_id in enumerate(rxn_id_list):
        rel_cov = K[i] / (rtkd * rtkd[i] + 1e-16)
        print(rxn_id)
        inds = np.argsort(np.abs(rel_cov))
        inds = np.flip(inds)[:nitems]
        print(rxn_id_arr[inds])
        print(rel_cov[inds])
        print(dy[inds])
        print()
    ana_set = {
        'K': K,
        'Kfull': gpr.K_,
        'y_pred': y_pred,
        'y': y,
        'alpha': alpha,
        'rxn_id_list': rxn_id_list,
    }
    with open('train_analysis.yaml', 'w') as f:
        yaml.dump(ana_set, f, Dumper=yaml.CDumper)

    dump(gpr, args.save_file)

    if args.mapped_fname is not None:
        for mfunc in mapping_plans:
            assert mfunc is not None
        dump(gpr.map(mapping_plans), args.mapped_fname)


if __name__ == '__main__':
    main()
