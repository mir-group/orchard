import numpy as np
from ciderpress.gpaw.gpaw_plot import *
from gpaw.sphere.lebedev import R_nv, weight_n, Y_nL
from ase.units import Bohr
from gpaw import restart as gpaw_restart
from ase import Atoms
from ase.data import chemical_symbols, ground_state_magnetic_moments
from orchard.workflow_utils import MLDFTDB_ROOT, VCML_ROOT

from collections import Counter
import os, sys, yaml

def get_atom_feat_wt(functional, a):
    ae_feat, ps_feat = functional.calculate_paw_feat_corrections_test(
        True, False, a=a, include_density=True)
    # check_feat is (...gL)
    for feat, sgn in [(ae_feat, 1), (ps_feat, -1)]:
        print(feat.shape, Y_nL.shape)
        feat = np.einsum('sigL,nL->sign', feat, Y_nL)
        nspin, nfeat = feat.shape[:2]
        dv_g = functional.setups[a].xc_correction.rgd.dv_g
        weight_gn = dv_g[:,None] * weight_n
        assert dv_g.size == feat.shape[-2]
        feat = feat.reshape(nspin, nfeat, -1)
        wt = weight_gn.reshape(-1)
        yield feat, sgn * wt

def get_features_and_weights(atoms, functional_init):
    functional = functional_init()
    atoms.calc.get_xc_difference(functional)
    vol = atoms.get_volume()
    ae_feat = functional.get_features_on_grid(True, include_density=True)
    nspin, nfeat = ae_feat.shape[:2]
    size = np.prod(ae_feat.shape[2:])
    all_wt = np.ones(size) / size * vol / Bohr**3
    all_wt = all_wt.flatten()
    ae_feat = ae_feat.reshape(nspin, nfeat, -1)
    cond = ae_feat[:,0,:].sum(axis=0) > 1e-6 # hopefully save disk space and avoid negative density
    ae_feat = ae_feat[...,cond]
    all_wt = all_wt[cond]
    assert all_wt.size == ae_feat.shape[-1]
    
    for a in range(len(atoms)):
        for ft, wt, in get_atom_feat_wt(functional, a):
            assert wt.shape[-1] == ft.shape[-1]
            ae_feat = np.append(ae_feat, ft, axis=-1)
            all_wt = np.append(all_wt, wt, axis=-1)
    
    ae_feat *= nspin
    ae_feat[:,1,:] *= nspin
    all_wt /= nspin
    print(ae_feat.shape, all_wt.shape)
    print(np.dot(ae_feat, all_wt))
    return ae_feat, all_wt

def get_sol62_data(MLDFTDB_ROOT, sysid, method, functional,
                   ref_method='NSCF_EXX', GEOM_FUNCTIONAL='PBE',
                   compute_ae_terms=False):
    data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method, sysid)
    atoms, calc = gpaw_restart(os.path.join(data_dir, 'calc.gpw'))
    ae_feat, all_wt = get_features_and_weights(atoms, functional)
    if ae_feat.shape[0] == 2:
        ae_feat = np.append(ae_feat[0], ae_feat[1], axis=-1)
        all_wt = np.append(all_wt, all_wt)
    else:
        ae_feat = ae_feat[0]
    
    cond = ae_feat[0] > 1e-6
    ae_feat = ae_feat[...,cond]
    all_wt = all_wt[...,cond]
    
    with open(os.path.join(data_dir, 'run_info.yaml'), 'r') as f:
        outdata = yaml.load(f, Loader=yaml.Loader)
    data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', ref_method, sysid)
    with open(os.path.join(data_dir, 'run_info.yaml'), 'r') as f:
        refdata = yaml.load(f, Loader=yaml.Loader)
    atoms = Atoms.fromdict(outdata['struct'])
    formula = Counter(atoms.get_atomic_numbers())
    ntot = 0
    ecoh = outdata['e_tot']
    ecoh_ref = refdata['e_tot']
    if compute_ae_terms:
      magmom_min = -1
      for Z, count in formula.items():
        ntot += count
        e_atom_min = 1e10
        magmom0 = int(ground_state_magnetic_moments[Z])
        for magmom in range(magmom0,magmom0+1):
            atom_id = 'SOL62/{}/atoms/{}_magmom{}'.format(
                GEOM_FUNCTIONAL,
                chemical_symbols[Z],
                magmom,#int(ground_state_magnetic_moments[Z]),
            )
            data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', ref_method, atom_id)
            loaddir = os.path.join(data_dir, 'run_info.yaml')
            print(loaddir, os.path.exists(loaddir))
            if os.path.exists(loaddir):
                print('FOUND MAGMOM', magmom)
                with open(loaddir, 'r') as f:
                    atom_outdata = yaml.load(f, Loader=yaml.Loader)
                if e_atom_min > atom_outdata['e_tot']:
                    e_atom_min = atom_outdata['e_tot']
                    magmom_min = magmom
            else:
                continue
        ecoh_ref -= count * e_atom_min
        atom_id = 'SOL62/{}/atoms/{}_magmom{}'.format(
            GEOM_FUNCTIONAL,
            chemical_symbols[Z],
            magmom_min,
        )
        data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method, atom_id)
        loaddir = os.path.join(data_dir, 'run_info.yaml')
        with open(loaddir, 'r') as f:
            ecoh -= count * yaml.load(f, Loader=yaml.Loader)['e_tot']
        atm, calc = gpaw_restart(os.path.join(data_dir, 'calc.gpw'))
        ae_feat_atm, all_wt_atm = get_features_and_weights(atm, functional)
        all_wt_atm *= -1 * count
        if ae_feat_atm.shape[0] == 2:
            ae_feat_atm = np.append(ae_feat_atm[0], ae_feat_atm[1], axis=-1)
        else:
            ae_feat_atm = ae_feat_atm[0]
        ae_feat = np.append(ae_feat, ae_feat_atm, axis=-1)
        all_wt = np.append(all_wt, all_wt_atm, axis=-1)
    if compute_ae_terms:
        all_wt /= ntot
        ecoh_ref /= ntot
    return ecoh_ref, ae_feat, all_wt
    

def save_dat(save_dir, ecoh_ref, ae_feat, all_wt):
    ref_vals = ecoh_ref/all_wt.sum() * np.ones(all_wt.size)
    files = ['val', 'wt', 'desc', 'rho']
    vecs = [ref_vals, all_wt, ae_feat, ae_feat[:1]]
    for fname, vec in zip(files, vecs):
        np.save(os.path.join(save_dir, fname + '.npy'), vec)

def get_sol62_results(sysid, method, GEOM_FUNCTIONAL='PBE'):
    data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method, sysid)
    with open(os.path.join(data_dir, 'run_info.yaml'), 'r') as f:
        outdata = yaml.load(f, Loader=yaml.Loader)
    atoms = Atoms.fromdict(outdata['struct'])
    formula = Counter(atoms.get_atomic_numbers())
    ntot = 0
    ecoh = outdata['e_tot']
    for Z, count in formula.items():
        ntot += count
        e_atom_min = 1e10
        for magmom in range(0,10):
            #if magmom != int(ground_state_magnetic_moments[Z]):
            #    continue
            atom_id = 'SOL62/{}/atoms/{}_magmom{}'.format(
                GEOM_FUNCTIONAL,
                chemical_symbols[Z],
                magmom,#int(ground_state_magnetic_moments[Z]),
            )
            data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method, atom_id)
            loaddir = os.path.join(data_dir, 'run_info.yaml')
            if os.path.exists(loaddir):
                with open(loaddir, 'r') as f:
                    atom_outdata = yaml.load(f, Loader=yaml.Loader)
                e_atom_min = min(e_atom_min, atom_outdata['e_tot'])
            else:
                continue
                #print('Missing atom gs', chemical_symbols[Z], ground_state_magnetic_moments[Z])
                #return 0
        ecoh -= count * e_atom_min
    return ecoh / ntot

def get_sol62_refvals():
    from orchard.vcml_data.eval import values as VCMLValues
    refvals = VCMLValues()
    pbevals = VCMLValues()
    refvals.get_values('REF')
    pbevals.get_values('PBE')
    return refvals, pbevals

def get_sol62_error(sysids, method_name, comp_method=None, GEOM_FUNCTIONAL='PBE'):
    ntot = 0
    errtot = 0
    errtot2 = 0
    pbetot = 0
    pbetot2 = 0
    refvals, pbevals = get_sol62_refvals()
    
    ecoh_dict = {}
    if comp_method is not None:
        ref_dict = {}
    ev_per_ha = 27.211399
    for struct_id, sysid in sysids:
        ecoh = get_sol62_results(sysid, method_name)
        ecoh_dict[struct_id.split('_')[0]] = ev_per_ha * ecoh
        if comp_method is not None:
            ecoh = get_sol62_results(sysid, comp_method)
            ref_dict[struct_id.split('_')[0]] = ev_per_ha * ecoh

    for form_id, val, pbeval in zip(refvals.ecohSys, refvals.ecoh, pbevals.ecoh):
        if form_id not in ecoh_dict:
            continue
        if comp_method is not None:
            val = ref_dict[form_id]
        if ecoh_dict[form_id] != 0:
            ntot += 1
            err = abs(val - ecoh_dict[form_id])
            perr = abs(val - pbeval)
            errtot += err
            pbetot += perr
            errtot2 += err**2
            pbetot2 += perr**2
        else:
            raise ValueError
    return ntot, errtot/ntot, (errtot2/ntot)**0.5

