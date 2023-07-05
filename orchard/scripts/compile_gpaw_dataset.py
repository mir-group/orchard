import logging
import os, yaml
from argparse import ArgumentParser
from ciderpress.density import LDA_FACTOR, GG_AMIN
from orchard.workflow_utils import SAVE_ROOT, load_mol_ids
from orchard.gpaw_tasks import StoreFeatures


def compile_dataset(
        DESC_NAME,
        DATASET_NAME,
        MOL_IDS,
        SAVE_ROOT,
        FUNCTIONAL,
        gg_kwargs,
        version='b',
        save_gap_data=False,
        save_baselines=True,
):
    if version not in ['b', 'd']:
        raise ValueError('Unsupported version for new dataset module')

    save_dir = os.path.join(
        SAVE_ROOT, 'DATASETS', FUNCTIONAL,
        version, DESC_NAME,
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    settings = {
        'DATASET_NAME': DATASET_NAME,
        'DESC_NAME': DESC_NAME,
        'MOL_IDS': MOL_IDS,
        'SAVE_ROOT': SAVE_ROOT,
        'FUNCTIONAL': FUNCTIONAL,
        'BASIS': 'GPAW',
        'version': version,
    }
    settings.update(gg_kwargs)
    print(save_dir, SAVE_ROOT, DESC_NAME)
    print(os.path.join(save_dir,
             '{}_settings.yaml'.format(DATASET_NAME)))
    with open(os.path.join(save_dir,
             '{}_settings.yaml'.format(DATASET_NAME)), 'w') as f:
        yaml.dump(settings, f)

    fwlist = {}

    for MOL_ID in MOL_IDS:
        logging.info('Computing descriptors for {}'.format(MOL_ID))
        save_file = os.path.join(save_dir, MOL_ID + '.hdf5')
        data_dir = os.path.join(SAVE_ROOT, 'PW-KS', FUNCTIONAL, MOL_ID)
        calc_settings = {
            'task': 'FEAT',
            'data_dir': data_dir,
            'save_file': save_file,
            'save_gap_data': save_gap_data,
            'save_baselines': save_baselines,
            'gg_kwargs': gg_kwargs,
            'version': version,
        }
        fwname = 'gpaw_feature_{}_{}'.format(version, MOL_ID)
        fwlist[fwname] = StoreFeatures(settings=calc_settings)

    return fwlist


def compile_exx_dataset(
        MOL_IDS,
        SAVE_ROOT,
        FUNCTIONAL,
        kpt_density,
        save_gap_data=False,
        save_baselines=True,
):
    fwlist = {}

    for MOL_ID in MOL_IDS:
        logging.info('Computing exx for {}'.format(MOL_ID))
        data_dir = os.path.join(SAVE_ROOT, 'PW-KS', FUNCTIONAL, MOL_ID)
        new_kpts = None if 'magmom' in MOL_ID else \
            {'density': kpt_density, 'even': True, 'gamma': True}
        calc_settings = {
            'task': 'EXX',
            'kpts': new_kpts,
            'encut': 520,
            'data_dir': data_dir,
            'save_gap_data': save_gap_data,
            'save_baselines': save_baselines,
        }
        fwname = 'gpaw_exx_{}'.format(MOL_ID)
        fwlist[fwname] = StoreFeatures(settings=calc_settings)

    return fwlist


def main():
    logging.basicConfig(level=logging.INFO)

    m_desc = 'Setup FWs to compile dataset of XC descriptors with GPAW'

    parser = ArgumentParser(description=m_desc)
    parser.add_argument('mol_id_file', type=str,
                        help='yaml file from which to read mol_ids to parse')
    parser.add_argument('--functional', metavar='functional', type=str, default=None,
                        help='exchange-correlation functional, HF for Hartree-Fock')
    parser.add_argument('--version', default='c', type=str,
                        help='version of descriptor set. Default c')
    parser.add_argument('--gg-a0', default=8.0, type=float)
    parser.add_argument('--gg-facmul', default=1.0, type=float)
    parser.add_argument('--gg-amin', default=GG_AMIN, type=float)
    parser.add_argument('--gg-vvmul', default=1.0, type=float,
                        help='For version b only, mul to get second coord exponent')
    parser.add_argument('--suffix', default=None, type=str,
                        help='customize data directories with this suffix')
    parser.add_argument('--save-gap-data', action='store_true')
    parser.add_argument('--exx-only', action='store_true')
    parser.add_argument('--kpt-density', default=4.5, type=float)
    args = parser.parse_args()

    version = args.version.lower()
    if version not in ['b', 'd']:
        raise ValueError('Unsupported descriptor set')

    mol_ids = load_mol_ids(args.mol_id_file)
    if args.mol_id_file.endswith('.yaml'):
        mol_id_code = args.mol_id_file[:-5]
    else:
        mol_id_code = args.mol_id_file
    gg_kwargs = {
        'amin': args.gg_amin,
        'a0': args.gg_a0,
        'fac_mul': args.gg_facmul
    }
    if version in ['b', 'd']:
        gg_kwargs['vvmul'] = args.gg_vvmul
    if args.exx_only:
        res = compile_exx_dataset(
            mol_ids,
            SAVE_ROOT,
            args.functional,
            kpt_density=args.kpt_density,
            save_gap_data=args.save_gap_data,
        )
    else:
        res = compile_dataset(
            '_UNNAMED' if args.suffix is None else args.suffix,
            mol_id_code.upper().split('/')[-1],
            mol_ids,
            SAVE_ROOT,
            args.functional,
            gg_kwargs,
            version=version,
            save_gap_data=args.save_gap_data,
        )
    from fireworks import LaunchPad, Firework
    launchpad = LaunchPad.auto_load()
    for fw in res:
        fw = Firework([res[fw]], name=fw)
        print(fw.name)
        launchpad.add_wf(fw)


if __name__ == '__main__':
    main()
