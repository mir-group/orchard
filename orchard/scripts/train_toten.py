from argparse import ArgumentParser
import os
import numpy as np
from joblib import dump, load
from orchard.workflow_utils import SAVE_ROOT, load_rxns
from mldftdat.models.gp import *
from mldftdat.models.compute_mol_cov import compute_tr_covs, compute_tr_covs_ex, \
                                            compute_heg_covs, compute_new_alpha, \
                                            reduce_model_size_
import yaml

def parse_settings(args):
    fname = args.datasets_list[0]
    if args.suffix is not None:
        fname = fname + '_' + args.suffix
    fname = os.path.join(SAVE_ROOT, 'DATASETS', args.functional,
                         args.basis, args.version, fname)
    print(fname)
    with open(os.path.join(fname, 'settings.yaml'), 'r') as f:
        d = yaml.load(f, Loader=yaml.Loader)
    args.gg_a0 = d.get('a0')
    args.gg_amin = d.get('amin')
    args.gg_facmul = d.get('fac_mul')

def parse_list(lststr, T=int):
    return [T(substr) for substr in lststr.split(',')]

def main():
    parser = ArgumentParser(description='Trains a GP exchange model')

    parser.add_argument('load_file', type=str, help='file from which to load GP')
    parser.add_argument('save_file', type=str, help='file to which to save new GP')
    parser.add_argument('reactions_list', nargs='+', help='dataset names for reactions files')
    parser.add_argument('basis', metavar='basis', type=str,
                        help='basis set code')
    parser.add_argument('--extra-datasets', nargs='+', default=None,
                        help='extra datasets needed for training reactions')
    parser.add_argument('--extra-dirs', nargs='+', default=None,
                        help='Extra dirs to search for datasets if not found in SAVE_ROOT')
    parser.add_argument('--functional', metavar='functional', type=str, default=None,
                        help='exchange-correlation functional, HF for Hartree-Fock')
    #parser.add_argument('-c', '--density-cutoff', type=float, default=1e-4)
    parser.add_argument('-s', '--seed', help='random seed', default=0, type=int)
    parser.add_argument('-d', '--delete-k', action='store_true',
                        help='Delete L (LL^T=K the kernel matrix) to save disk space. Need to refit when reloading to calculate covariance.')
    parser.add_argument('-o', '--desc-order', default=None,
                        help='comma-separated list of descriptor order with no spaces. must start with 0,1.')
    parser.add_argument('-x', '--xed-y-code', default='CHACHIYO', type=str)
    parser.add_argument('-v', '--version', default='c', type=str,
                        help='version of descriptor set. Default c')
    parser.add_argument('--suffix', default=None, type=str,
                        help='customize data directories with this suffix')
    parser.add_argument('--train-to-ae', action='store_true',
                        help='Train to atomization exchange energy instead of total exchange energy')
    parser.add_argument('--maxz-toten', type=int, default=18,
                        help='Train all atoms with Z>maxz_toten to energy differences rather than total energy')
    parser.add_argument('--maxz-mul', type=int, default=18)
    parser.add_argument('--atom-mul', type=float, default=1.0)
    parser.add_argument('--mol-mul', type=float, default=1.0)
    parser.add_argument('--fit-ae-only', action='store_true')
    parser.add_argument('--fix-fxsigma-to-molsigma', action='store_true')
    parser.add_argument('--nmax-sparse', type=int, default=None, help='If set, not more than this many points used in sparse set')
    parser.add_argument('--control-tol', type=float, default=-1e-5, help='Reduce control point size for given tol, negative means ignore, only allowed when fit-ae-only is true')
    parser.add_argument('--mol-sigma', type=float, default=np.sqrt(1e-5), help='Standard deviation noise parameter for total molecular energy data')
    parser.add_argument('--per-atom-sigma', type=float, default=0.0, help='Standard deviation noise parameter added per atom for total molecular energy data, excluding the first atom')
    parser.add_argument('--solids-dir', type=str, default=None, help='Read a solids DB from here')
    parser.add_argument('--solids-subset', type=str, default=None, help='Filename for subset ids for solids db')
    parser.add_argument('--skip-freq', type=int, default=0, help='Reduce number of xed training points by factor of skip_freq')
    parser.add_argument('--mol-heg', action='store_true', help='Include HEG constraint in molecules dataset')
    args = parser.parse_args()

    #parse_settings(args)

    np.random.seed(args.seed)
    model = load(args.load_file)
    args.datasets_list = model.args.datasets_list[::2]
    if args.extra_datasets is not None:
        for d in args.extra_datasets:
            args.datasets_list.append(d)
    train_to_ae = args.train_to_ae
    print(args.load_file, args.save_file, args.datasets_list)

    if args.control_tol > 0:
        #if not args.fit_ae_only:
        #    raise NotImplementedError('No XED training + control_tol yet')
        model = reduce_model_size_(model, args.control_tol, args.nmax_sparse)
    if args.fix_fxsigma_to_molsigma:
        from sklearn.gaussian_process.kernels import WhiteKernel
        assert isinstance(model.gp.kernel_.k2.k1, WhiteKernel)
        model.gp.kernel_.k2.k1.set_params(noise_level=args.mol_sigma**2)

    assert len(args.datasets_list) != 0, 'Need training data'
    nd = len(args.datasets_list)

    if model.args.use_ex_kernel:
        get_covs = compute_tr_covs_ex
    else:
        get_covs = compute_tr_covs

    vwrtt_list, exx_list = [], []

    rxn_list = []
    for rxn_id in args.reactions_list:
        rxn_dict = load_rxns(rxn_id)
        for v in list(rxn_dict.values()):
            rxn_list.append(v)

    import yaml
    system_ids = []
    for i in range(nd):
        fname = args.datasets_list[i]
        if args.suffix is not None:
            fname = fname + '_' + args.suffix
        if args.extra_dirs is None:
            fname = os.path.join(SAVE_ROOT, 'DATASETS', args.functional,
                                 args.basis, args.version, fname)
        else:
            ddirs = [SAVE_ROOT] + args.extra_dirs
            for dd in ddirs:
                cdd = os.path.join(dd, 'DATASETS', args.functional,
                                   args.basis, args.version, fname)
                print(cdd)
                if os.path.exists(cdd):
                    fname = cdd
                    break
            else:
                raise FileNotFoundError('Could not find dataset in provided dirs')
        vwrtt_mat, exx = get_covs(model, fname)
        vwrtt_list.append(vwrtt_mat)
        exx_list.append(exx)
        fname = os.path.join(fname, 'settings.yaml')
        with open(fname, 'r') as f:
            settings = yaml.load(f,Loader=yaml.Loader)
            system_ids += settings['MOL_IDS']
    if args.mol_heg:
        vw_tmp, exx_tmp = compute_heg_covs(model)
        vwrtt_list.append(vw_tmp)
        exx_list.append(exx_tmp)
        system_ids.append('UNIFORM_ELECTRON_GAS')
        rxn_list.append({'structs': ['UNIFORM_ELECTRON_GAS'], 'counts': [1], 'noise': 0.00})
    vwrtt_mat = np.hstack(vwrtt_list)
    exx = np.concatenate(exx_list)
    idmap = {}
    for ind, sysid in enumerate(system_ids):
        idmap[sysid] = ind

    print('IDMAP')
    for k, v in idmap.items():
        print('k v', k, v)
    vwrtt_rxns = []
    exx_rxns = []
    noises = []
    for rxn in rxn_list:
        vw = 0
        ex = 0
        na = 0
        for sysid, count in zip(rxn['structs'], rxn['counts']):
            try:
                ind = idmap[sysid]
            except KeyError:
                raise RuntimeError('Datasets must contain all system ids in reaction sets')
            na += abs(count)
            vw += count * vwrtt_mat[:,ind]
            ex += count * exx[ind]
        vwrtt_rxns.append(vw)
        exx_rxns.append(ex)
        if rxn.get('noise') is not None:
            print('Fixed noise', rxn.get('noise'))
            noises.append(rxn['noise'])
        elif rxn.get('noise_factor') is not None:
            noises.append(rxn['noise_factor'] * args.mol_sigma)
        else:
            noises.append(args.mol_sigma)
    vwrtt_rxns = np.array(vwrtt_rxns, dtype=np.float64).T
    exx_rxns = np.array(exx_rxns, dtype=np.float64)
    noise_list = np.array(noises, dtype=np.float64)

    """
    if args.solids_dir:
        import ase.io.vasp
        from ase.data import ground_state_magnetic_moments, chemical_symbols
        def _load_formula(name):
            #from orchard.workflow_utils import VCML_ROOT
            VCML_ROOT = '/n/holystore01/LABS/kozinsky_lab/Lab/User/kbystrom/vcml_data'
            if 'magmom' not in name:
                name = name + '/volume_0'
            rdir = os.path.join(VCML_ROOT, 'datasets/inout_vasp/SOL62/PBE/SOL62', name)
            atoms = ase.io.vasp.read_vasp(os.path.join(rdir, 'POSCAR'))
            nums = atoms.get_atomic_numbers()
            return Counter(nums)
        from collections import Counter
        ddir = os.path.join(args.solids_dir, args.suffix)
        dirs = next(os.walk(ddir))[1]
        if args.solids_subset is not None:
            with open(args.solids_subset, 'r') as f:
                subset = [l.strip() for l in f.readlines()]
                dirs = [d for d in dirs if d in subset]
        print(dirs)
        sol_dirs = [d for d in dirs if 'magmom' not in d]
        atom_dirs = [d for d in dirs if 'magmom' in d]
        sol_data = {}
        sol_vw = []
        sol_exx = []
        for d in dirs:
            print('STARTING', d)
            sol_data[d] = list(get_covs(model, os.path.join(ddir, d), unit='eV'))
            print(sol_data[d][0].shape, sol_data[d][1].shape)
        for sol_d in sol_dirs:
            formula = _load_formula(sol_d)
            tot = 0
            for Z, count in formula.items():
                tot += count
                magmom = ground_state_magnetic_moments[Z]
                el = chemical_symbols[Z]
                atom_id = '{}_magmom{}'.format(el, int(magmom))
                sol_data[sol_d][0] -= count * sol_data[atom_id][0]
                sol_data[sol_d][1] -= count * sol_data[atom_id][1]
            sol_vw.append(sol_data[sol_d][0][:,0] / tot)
            sol_exx.append(sol_data[sol_d][1][0] / tot)
    """

    vwrtt_mat = vwrtt_rxns
    exx = exx_rxns
    """
    if args.solids_dir:
        sol_exx = np.array(sol_exx)
        sol_vw = np.array(sol_vw).T
        print(exx.shape)
        print(sol_exx.shape)
        print(sol_vw.shape)
        print(vwrtt_mat.shape)
        exx = np.append(exx, sol_exx)
        vwrtt_mat = np.append(vwrtt_mat, sol_vw, axis=-1)
        noise_list = np.append(noise_list, 0.01 * np.ones(exx.size - noise_list.size))
    """

    frac = 1.0
    version = 7
    if model.args.use_ex_kernel:
        version = 8
    if args.fit_ae_only:
        version = 9
    print('SHAPES', vwrtt_mat.shape, exx.shape, noise_list.shape)
    alpha_new = compute_new_alpha(
        model, frac*vwrtt_mat, frac*exx,
        version=version, rho=True, molsigma=noise_list,
        skip_freq=args.skip_freq,
    )
    model.gp.alpha_ = alpha_new

    if args.delete_k:
        model.gp.L_ = None

    dump(model, args.save_file)

    """
    for i in range(nd):
        fname = args.datasets_list[i]
        if args.suffix is not None:
            fname = fname + '_' + args.suffix
        fname = os.path.join(SAVE_ROOT, 'DATASETS', args.functional,
                             args.basis, args.version, fname)
        get_covs(model, fname)
    """

if __name__ == '__main__':
    main()
