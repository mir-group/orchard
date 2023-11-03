from argparse import ArgumentParser
import os
import numpy as np
from joblib import load, dump
from orchard.workflow_utils import SAVE_ROOT, load_rxns
from ciderpress.models.train import DescParams, MOLGP
from ciderpress.models.dft_kernel import DFTKernel
from ciderpress.models.baselines import BASELINE_CODES
from ciderpress.xcutil.transform_data import FeatureList
from ciderpress.density import LDA_FACTOR
from ciderpress.new_dft.settings import EmptySettings, FeatureSettings
from pyscf.lib import chkfile
import importlib
import yaml

import traceback
import warnings
import sys


"""
Dataset file format:
systems:
  <dataset_name>:
    inverse_sampling_density: <integer>
    load_orbs: <bool>
  ...
reactions:
  <rxn_dataset_name0>: <integer 0 (X), 1 (C), or 2 (XC)>
  <rxn_dataset_name1>: <integer 0 (X), 1 (C), or 2 (XC)>
  ...
"""


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


def _get_name_dict(args):
    return {
        'SL'   : args.sl_feat_name,
        'NLDF' : args.nldf_feat_name,
        'NLOF' : args.nlof_feat_name,
        'SDMX' : args.sdmx_feat_name,
        'HYB'  : args.hyb_feat_name,
    }


def parse_settings(args):
    set0 = args.datasets_list[0]
    with open(args.normalizer_file, 'r') as f:
        normalizers = yaml.load(f, Loader=yaml.CLoader)
    base_dname = os.path.join(
        SAVE_ROOT, 'DATASETS', args.functional, args.basis
    )
    settings_dict = {}
    name_dict = _get_name_dict(args)
    for feat_type, feat_name in name_dict.items():
        if feat_name is None:
            settings_dict[feat_type] = None
        dname = os.path.join(
            base_dname, feat_type, feat_name, set0
        )
        fname = os.path.join(dname, '{}_settings.yaml'.format(set0))
        with open(fname, 'r') as f:
            settings_dict[feat_type] = yaml.load(f, Loader=yaml.Loader)
    settings = FeatureSettings(
        sl_settings=settings_dict['SL'],
        nldf_settings=settings_dict['NLDF'],
        nlof_settings=settings_dict['NLOF'],
        sadm_settings=settings_dict['SDMX'],
        hyb_settings=settings_dict['HYB'],
        normalizers=normalizers,
    )
    return settings


def find_datasets(fname, args):
    name_dict = _get_name_dict(args)
    ddirs = {}
    for feat_type, feat_name in name_dict.items():
        fname = '{}_settings.yaml'.format(fname)
        reldir = os.path.join(
            'DATASETS', args.functional, args.basis,
            feat_type, feat_name, fname
        )
        if args.extra_dirs is None:
            fname = os.path.join(SAVE_ROOT, reldir)
        else:
            ddirs = [SAVE_ROOT] + args.extra_dirs
            for dd in ddirs:
                cdd = os.path.join(dd, 'DATASETS', reldir)
                print(cdd)
                if os.path.exists(cdd):
                    fname = cdd
                    break
            else:
                raise FileNotFoundError('Could not find dataset in provided dirs')
        ddirs[feat_type] = os.path.dirname(fname)
    return ddirs


def get_plan_module(plan_file):
    if plan_file.startswith('@'):
        plan_module = importlib.import_module(plan_file[1:])
    else:
        assert os.path.exists(plan_file)
        spec = importlib.util.spec_from_file_location('plan_module', plan_file)
        plan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan_module)
    return plan_module


def parse_dataset_for_ctrl(fname, n, args):
    dirname = find_dataset(fname, args)
    with open(os.path.join(dirname, '{}_settings.yaml'.format(fname)), 'r') as f:
        settings = yaml.load(f, Loader=yaml.CLoader)
        mol_ids = settings['MOL_IDS']
    Xlist = []
    ylist = []
    for mol_id in mol_ids:
        fname = os.path.join(dirname, mol_id + '.hdf5')
        data = chkfile.load(fname, 'train_data')
        cond = data['desc'][:, 0, :] > args.density_cutoff
        print(data['desc'].shape, data['val'].shape)
        y = data['val'][cond] / (LDA_FACTOR * data['desc'][:, 0][cond]**(4.0 / 3)) - 1
        cond = np.all(cond, axis=0)
        X = data['desc'][:, :, cond]
        if args.randomize:
            inds = np.arange(X.shape[-1])
            np.random.shuffle(inds)
            X = X[..., inds]
        if n is not None:
            Xlist.append(X[..., ::n])
            ylist.append(y[::n])
    return Xlist, ylist, mol_ids


def main():
    parser = ArgumentParser(
        description="Fits a GP exchange(-correlation) model to "
                    "molecular/solid-state energy differences and "
                    "orbital energies."
    )

    parser.add_argument(
        'save_file', type=str, help='file to which to save new GP'
    )
    parser.add_argument(
        'ref-feat-name', type=str, help='Name of reference data set'
    )
    parser.add_argument(
        'sl-feat-name', type=str, help='Name of semilocal feature set'
    )
    parser.add_argument(
        'normalizer-file', type=str, help='Path to normalizer yaml file.'
    )
    parser.add_argument(
        'basis', metavar='basis', type=str, help='basis set code'
    )
    parser.add_argument(
        '--nldf-feat-name', type=str, default=None,
        help='Name of NLDF feature set.'
    )
    parser.add_argument(
        '--nlof-feat-name', type=str, default=None,
        help='Name of NLOF feature set.'
    )
    parser.add_argument(
        '--sdmx-feat-name', type=str, default=None,
        help='Name of SDMX feature set.'
    )
    parser.add_argument(
        '--hyb-feat-name', type=str, default=None,
        help='Name of hybrid DFT feature set.'
    )
    parser.add_argument(
        '--kernel-plan-file', type=str,
        help='Settings file for list of kernels. See '
             'ciderpress.models.kernel_plans.settings_example.yaml '
             'for documentation and format.'
    )
    parser.add_argument(
        '--dataset-file', type=str,
        help='Path to yaml file containing names of datasets to load '
             'along with instructions for loading said datasets.'
    )
    parser.add_argument(
        '--datasets-list', nargs='+',
        help='Pairs of dataset names and inverse sampling densities.'
    )
    parser.add_argument(
        '--load-orbs-list', nargs='+',
        help='Pairs of dataset names and 0 (n0) or 1 (yes) for '
             'whether to load orbital occupation gradients. Default is 1 if '
             'orbs are in the dataset, else 0. This string overrides defaults.'
    )
    parser.add_argument(
        '--reactions-list', nargs='+',
        help='Pairs of dataset names and modes for reactions files; '
             'mode 0=x-only, 1=c-only, 2-xc'
    )
    parser.add_argument(
        '--extra-dirs', nargs='+', default=None,
        help='Extra dirs to search for datasets if not found in SAVE_ROOT.'
    )
    parser.add_argument(
        '--functional', type=str, default=None,
        help='XC functional for reference data, HF for Hartree-Fock.'
    )
    parser.add_argument('-c', '--density-cutoff', type=float, default=1e-6)
    parser.add_argument('-s', '--seed', help='random seed', default=0, type=int)
    parser.add_argument(
        '-d', '--delete-k', action='store_true',
        help='Delete L (LL^T=K the kernel matrix) to save disk '
             'space. Need to refit when reloading to calculate '
             'covariance.'
    )
    parser.add_argument(
        '--nmax-sparse', type=int, default=None,
        help='If set, not more than this many points used in sparse set.'
    )
    parser.add_argument(
        '--control-tol', type=float, default=-1e-5,
        help='Reduce control point size for given tol. '
             'Negative value means to ignore.'
    )
    parser.add_argument(
        '--mol-sigma', tpe=float, default=0.03,
        help='Standard deviation noise parameter for total molecular energy data.'
    )
    parser.add_argument('--scale-override', type=float, default=None)
    parser.add_argument('--scale-mul', type=float, default=1.0)
    parser.add_argument(
        '--length-scale-mul', type=float, nargs='+', default=[1.0],
        help='Used for automatic length-scale initial guess.'
    )
    parser.add_argument(
        '--min-lscale', type=float, default=None,
        help='Minimum length-scale for GP kernel.'
    )
    parser.add_argument(
        '--libxc-baseline', type=str, default=None,
        help='Baseline libxc functional for the full model'
    )
    parser.add_argument(
        '--mapped-fname', type=str, default=None,
        help='If not None, map model and same to this file.'
    )
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument(
        '--debug-model', type=str, default=None,
        help='Load joblib and print debug'
    )
    parser.add_argument(
        '--debug-spline', type=str, default=None,
        help='Load joblib and print debug'
    )
    args = parser.parse_args()
    settings = parse_settings(args)
    if args.debug_model is not None:
        args.debug_model = load(args.debug_model)
    if args.debug_spline is not None:
        args.debug_spline = load(args.debug_spline)

    with open(args.dataset_file, 'r') as f:
        data_settings = yaml.load(f, Loader=yaml.CLoader)
    with open(args.kernel_plan_file, 'r') as f:
        kernel_plans = yaml.load(f, Loader=yaml.CLoader)

    datasets_list = list(data_settings['systems'].keys())

    Xlist = []
    ylist = []
    molid_map = {}
    for dset_name in datasets_list:
        n = data_settings['systems'][dset_name].get('inverse_sampling_density')
        Xlist_tmp, y_tmp, dset_ids = parse_dataset_for_ctrl(
            dset_name, n, args
        )
        Xlist += Xlist_tmp
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
            feature_list,
            plan['mode'],
            BASELINE_CODES[plan['multiplicative_baseline']],
            additive_baseline=BASELINE_CODES.get(plan['additive_baseline']),
            ctrl_tol=ctrl_tol,
            ctrl_nmax=ctrl_nmax,
        ))
        X1 = kernels[-1].X0Tlist_to_X1array(Xlist)
        print('SHAPES', X1.shape, yctrl.shape)
        kernel = plan_module.get_kernel(
            natural_scale=np.var(yctrl) if args.scale_override is None else args.scale_override,
            natural_lscale=np.std(X1, axis=0),
            scale_factor=args.scale_mul,
            lscale_factor=args.length_scale_mul,
        )
        kernels[-1].set_kernel(kernel)
        if 'mapping_plan' in dir(plan_module):
            mfunc = plan_module.mapping_plan
        else:
            mfunc = None
        mapping_plans.append(mfunc)

    gpr = MOLGP(
        kernels,
        settings,
        libxc_baseline=args.libxc_baseline,
        default_noise=args.mol_sigma,
    )
    gpr.args = args

    gpr.set_control_points(Xlist, reduce=True)

    rxn_list = []
    rxn_ids = list(data_settings['reactions'].keys())
    for i, rxn_id in enumerate(rxn_ids):
        rxn_dict = load_rxns(rxn_id)
        mode = data_settings['reactions'][rxn_id].get('mode') or 0
        for v in list(rxn_dict.values()):
            rxn_list.append((mode, v))

    for i, fname in enumerate(datasets_list):
        load_orbs = data_settings['systems'][fname].get('load_orbs')
        mol_ids = molid_map[fname]
        fnames = find_datasets(fname, args)
        gpr.store_mol_covs(
            fnames, mol_ids, get_orb_deriv=load_orbs, get_correlation=True
        )

    gpr.add_reactions(rxn_list)

    gpr.fit()

    dump(gpr, args.save_file)

    if args.mapped_fname is not None:
        for mfunc in mapping_plans:
            assert mfunc is not None
        dump(gpr.map(mapping_plans), args.mapped_fname)


if __name__ == '__main__':
    main()
