import os
from orchard.pyscf_tasks import StoreFeatures2
from orchard.workflow_utils import get_save_dir, SAVE_ROOT, load_mol_ids
from ciderpress.pyscf.analyzers import ElectronAnalyzer, RHFAnalyzer, UHFAnalyzer
from ciderpress.new_dft.settings import SemilocalSettings, NLDFSettings, \
    FracLaplSettings, SDMXBaseSettings, HybridSettings
from ciderpress.pyscf.descriptors import get_descriptors
from pyscf.lib import chkfile
import numpy as np
import logging
import yaml
import time
from argparse import ArgumentParser


def intk_to_strk(d):
    if not isinstance(d, dict):
        return d
    nd = {}
    for k, v in d.items():
        nd[str(k)] = intk_to_strk(v)
    return nd


def get_feat_type(settings):
    if settings == 'l':
        return 'REF'
    elif isinstance(settings, SemilocalSettings):
        return 'SL'
    elif isinstance(settings, NLDFSettings):
        return 'NLDF'
    elif isinstance(settings, FracLaplSettings):
        return 'NLOF'
    elif isinstance(settings, SDMXBaseSettings):
        return 'SDMX'
    elif isinstance(settings, HybridSettings):
        return 'HYB'
    else:
        raise ValueError


def compile_single_system(
        settings, save_file, analyzer_file, sparse_level, orbs, save_baselines
):
    start = time.monotonic()
    analyzer = ElectronAnalyzer.load(analyzer_file)
    if sparse_level is not None:
        old_analyzer = analyzer
        Analyzer = UHFAnalyzer if analyzer.atype == 'UHF' else RHFAnalyzer
        analyzer = Analyzer(analyzer.mol, analyzer.dm, grids_level=sparse_level,
                            mo_occ=old_analyzer.mo_occ,
                            mo_coeff=old_analyzer.mo_coeff,
                            mo_energy=old_analyzer.mo_energy)
        if 'e_tot_orig' in old_analyzer._data:
            analyzer._data['xc_orig'] = old_analyzer.get('xc_orig')
            analyzer._data['exc_orig'] = old_analyzer.get('exc_orig')
            analyzer._data['e_tot_orig'] = old_analyzer.get('e_tot_orig')
        analyzer.perform_full_analysis()
    else:
        analyzer.get_rho_data()
    end = time.monotonic()
    logging.info('Analyzer load time {}'.format(end - start))

    start = time.monotonic()
    desc = get_descriptors(analyzer, settings, orbs=orbs)
    if orbs is not None:
        desc, ddesc, eigvals = desc
    else:
        ddesc = None
        eigvals = None
    end = time.monotonic()
    logging.info('Get descriptor time {}'.format(end - start))

    if isinstance(analyzer, UHFAnalyzer):
        spinpol = True
    else:
        spinpol = False
    if settings == 'l':
        values = analyzer.get('ex_energy_density')
        weights = analyzer.grids.weights
        coords = analyzer.grids.coords
        if spinpol:
            values = np.stack([values[0], values[1]])
        else:
            values = values[np.newaxis, :]
        data = {
            'coord' : coords,
            'nspin' : 2 if spinpol else 1,
            'rho_data' : desc,
            'val' : values,
            'wt' : weights
        }
        if orbs is not None:
            data['dval'] = intk_to_strk(analyzer.calculate_vxc_on_mo('HF', orbs))
            data['drho_data'] = intk_to_strk(ddesc)
            data['eigvals'] = intk_to_strk(eigvals)
        if save_baselines:
            data['xc_orig'] = analyzer.get('xc_orig')
            data['exc_orig'] = analyzer.get('exc_orig')
            data['e_tot_orig'] = analyzer.get('e_tot_orig')
    else:
        data = {
            'desc' : desc,
        }
        if orbs is not None:
            data['ddesc'] = intk_to_strk(ddesc)
    dirname = os.path.dirname(os.path.abspath(save_file))
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    chkfile.dump(save_file, 'train_data', data)


def compile_dataset(
        feat_settings,
        feat_name,
        dataset_name,
        mol_id_list,
        save_root,
        functional,
        basis,
        sparse_level=None,
        analysis_level=1,
        save_gap_data=False,
        save_baselines=True,
        make_fws=False,
        skip_existing=False,
        save_dir=None,
):
    if save_gap_data:
        orbs = {'O': [0], 'U': [0]}
    else:
        orbs = None
    feat_type = get_feat_type(feat_settings)
    if save_dir is None:
        save_dir = os.path.join(
            save_root, 'DATASETS', functional,
            basis, feat_type, feat_name
        )
    else:
        save_dir = os.path.join(save_dir, feat_type, feat_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    settings = {
        'DATASET_NAME'  : dataset_name,
        'FEAT_NAME'     : feat_name,
        'MOL_IDS'       : mol_id_list,
        'SAVE_ROOT'     : save_root,
        'FUNCTIONAL'    : functional,
        'BASIS'         : basis,
        'FEAT_SETTINGS' : feat_settings,
    }
    print(save_dir, save_root, feat_name)
    settings_fname = '{}_settings.yaml'.format(dataset_name)
    print(os.path.join(save_dir, settings_fname))
    with open(os.path.join(save_dir, settings_fname), 'w') as f:
        yaml.dump(settings, f)

    fwlist = {}
    for mol_id in mol_id_list:
        logging.info('Computing descriptors for {}'.format(mol_id))
        data_dir = get_save_dir(save_root, 'KS', basis, mol_id, functional)
        save_file = os.path.join(save_dir, mol_id + '.hdf5')
        if os.path.exists(save_file) and skip_existing:
            print('Already exists, skipping:', mol_id)
            continue
        analyzer_file = data_dir + '/analysis_L{}.hdf5'.format(analysis_level)
        args = [feat_settings, save_file, analyzer_file,
                sparse_level, orbs, save_baselines]
        if make_fws:
            fwname = 'feature_{}_{}'.format(feat_name, mol_id)
            args[0] = yaml.dump(args[0], Dumper=yaml.CDumper)
            fwlist[fwname] = StoreFeatures2(args=args)
        else:
            compile_single_system(*args)
    return fwlist


def main():
    logging.basicConfig(level=logging.INFO)
    m_desc = 'Compile dataset of XC descriptors'
    parser = ArgumentParser(description=m_desc)
    parser.add_argument(
        'mol_id_file', type=str,
        help='yaml file from whcih to read mol_ids to parse'
    )
    parser.add_argument(
        'feat_name', type=str,
        help='Name of the feature set being generated, used to make '
             'save directory for generated data.'
    )
    parser.add_argument(
        'basis', metavar='basis', type=str,
        help='Basis set that was used for the DFT calculations'
    )
    parser.add_argument(
        '--settings-file', metavar='settings_file', type=str, default=None,
        help="Path to a yaml file containing a serialized FeatureSettings "
             "class. If not provided, generates the reference data "
             "(i.e. semilocal density, EXX and XC reference, etc.)"
    )
    parser.add_argument(
        '--functional', metavar='functional', type=str, default=None,
        help='exchange-correlation functional, HF for Hartree-Fock'
    )
    parser.add_argument(
        '--analysis-level', default=1, type=int,
        help='Level of analysis to search for each system, looks '
             'for analysis_L{analysis-level}.hdf5'
    )
    parser.add_argument(
        '--sparse-grid', default=None, type=int, nargs='+',
        help='use a sparse grid to compute features, etc. '
             'If set, recomputes data.'
    )
    parser.add_argument(
        '--make-fws', action='store_true',
        help='If True, make a firework to generate features for each'
             'molecule, to be run later. If False, generate features'
             'for each molecule serially within this script.'
    )
    parser.add_argument(
        '--save-gap-data', action='store_true',
        help='If True, store the band gap data for eac molecule.'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='skip system if save_file exists already'
    )
    parser.add_argument(
        '--save-dir', default=None, type=str,
        help='override default save directory for features'
    )
    args = parser.parse_args()

    if args.settings_file is None or args.settings_file == '__REF__':
        settings = 'l'
    else:
        with open(args.settings_file, 'r') as f:
            settings = yaml.load(f, Loader=yaml.CLoader)

    mol_id_list = load_mol_ids(args.mol_id_file)
    if args.mol_id_file.endswith('.yaml'):
        mol_id_code = args.mol_id_file[:-5]
    else:
        mol_id_code = args.mol_id_file

    if args.sparse_grid is None:
        sparse_level = None
    elif len(args.sparse_grid) == 1:
        sparse_level = args.sparse_grid[0]
    elif len(args.sparse_grid) == 2:
        sparse_level = (args.sparse_grid[0], args.sparse_grid[1])
    else:
        raise ValueError('Sparse grid must be 1 or 2 integers')

    res = compile_dataset(
        settings,
        args.feat_name,
        mol_id_code.upper().split('/')[-1],
        mol_id_list,
        SAVE_ROOT,
        args.functional,
        args.basis,
        sparse_level=sparse_level,
        save_gap_data=args.save_gap_data,
        make_fws=args.make_fws,
        skip_existing=args.skip_existing,
        save_dir=args.save_dir
    )
    if args.make_fws:
        from fireworks import LaunchPad, Firework
        launchpad = LaunchPad.auto_load()
        for fw in res:
            fw = Firework([res[fw]], name=fw)
            print(fw.name)
            launchpad.add_wf(fw)


if __name__ == '__main__':
    main()
