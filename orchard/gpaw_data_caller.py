import yaml
import os
import sys
import numpy as np
from gpaw import restart, PW
from ase.parallel import paropen
from pyscf.lib import chkfile
from ase.units import Ha


def get_exx(data_dir, calc, kpts,
            save_gap_data=False):
    """
    :param save_dir:
    :param calc:
    :param p_be: (p_vbm, p_cbm), (s, k, n) for each
    :return:
    """
    from gpaw.hybrids.energy import non_self_consistent_energy
    if kpts is not None:
        calc.set(kpts=kpts)
    solver = calc.parameters.eigensolver
    #if solver is None:
    #    pass
    #elif not isinstance(solver, str) or solver.lower() == 'cg':
    #    calc.set(parallel={'domain': 1, 'band': 1})
    calc.get_potential_energy()
    if save_gap_data:
        from ase.dft.bandgap import bandgap
        gap, p_vbm, p_cbm = bandgap(calc)
        p_be = (p_vbm, p_cbm)
        print(p_be)
    else:
        p_be = None
    eterms = non_self_consistent_energy(calc, 'EXX')
    data = {}
    data['kpts'] = calc.parameters.kpts
    data['e_tot_orig'] = eterms[0] / Ha
    data['exc_orig'] = eterms[1] / Ha
    data['xc_orig'] = calc.hamiltonian.xc.name
    data['exx'] = eterms[3:].sum() / Ha
    if p_be is not None:
        from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues as nsceigs
        eig_dft_dict = {k : {} for k in ['O', 'U']}
        vxc_dft_dict = {k : {} for k in ['O', 'U']}
        vxc_hyb_dict = {k : {} for k in ['O', 'U']}
        for l, p in zip(['O', 'U'], p_be):
            eig_dft, vxc_dft, vxc_hyb = nsceigs(
                calc, 'EXX', n1=p[2], n2=p[2] + 1, kpt_indices=[p[1]]
            )
            eig_dft_dict[l][0] = eig_dft[p[0], 0, 0] / Ha
            vxc_dft_dict[l][0] = vxc_dft[p[0], 0, 0] / Ha
            vxc_hyb_dict[l][0] = vxc_hyb[p[0], 0, 0] / Ha
        data['eigvals'] = eig_dft_dict
        data['vxc_dft'] = vxc_dft_dict
        data['dval'] = vxc_hyb_dict
        data['p_be'] = p_be
    with paropen(os.path.join(data_dir, 'exx_data.yaml'), 'w') as f:
        yaml.dump(data, f, Dumper=yaml.CDumper)


def arr_to_strk(arr, nspin, p_be):
    if nspin == 2:
        v, c = (p_be[0][0], arr[0]), (p_be[1][0], arr[1])
    else:
        v, c = arr[0], arr[1]
    return {
        'O': {'0': v},
        'U': {'0': c},
    }

def intk_to_strk(d):
    if not isinstance(d, dict):
        return d
    nd = {}
    for k, v in d.items():
        nd[str(k)] = intk_to_strk(v)
    return nd



def save_features(save_file, data_dir, calc, version, gg_kwargs,
                  save_gap_data=False):
    from ciderpress.gpaw.analysis import get_features
    print('START FEAT')
    with paropen(os.path.join(data_dir, 'exx_data.yaml'), 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
    data.pop('kpts')
    if save_gap_data:
        data['eigvals'] = intk_to_strk(data['eigvals'])
        data['vxc_dft'] = intk_to_strk(data['vxc_dft'])
        data['dval'] = intk_to_strk(data['dval'])
        p_be = data.pop('p_be')
    else:
        if 'p_be' in data.keys():
            data.pop('p_be')
        p_be = None

    res = get_features(calc, p_i=p_be, version=version, **gg_kwargs)
    rho_res = get_features(calc, p_i=p_be, version='l')
    if p_be is None:
        feat_sig, all_wt = res
        rho_sig, _ = rho_res
    else:
        feat_sig, dfeat_jig, all_wt = res
        rho_sig, drho_jig, _ = rho_res
        data.update({
            'ddesc': arr_to_strk(dfeat_jig, feat_sig.shape[0], p_be),
            'drho_data': arr_to_strk(drho_jig, feat_sig.shape[0], p_be)
        })
    nspin = feat_sig.shape[0]
    data.update({
        'rho': rho_sig,
        'desc': feat_sig,
        'wt': all_wt,
        'nspin': nspin,
    })
    print('WRITE FEAT')
    data['val'] = data['exx'] * all_wt / (nspin * all_wt.sum())
    if nspin == 2:
        data['val'] = np.stack([data['val'], data['val']]) # sums to exx
    else:
        data['val'] = data['val'][np.newaxis, :]
    if calc.world.rank == 0:
        save_dir = os.path.dirname(os.path.abspath(save_file))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        chkfile.dump(save_file, 'train_data', data)


def call_gpaw():
    print('CALL GPAW')
    with paropen(sys.argv[1], 'r') as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    data_dir = settings['data_dir']
    task = settings['task'] # should be EXX or FEAT
    atoms, calc = restart(os.path.join(data_dir, 'calc.gpw'), txt='-')
    if task == 'EXX':
        get_exx(data_dir, calc, settings['kpts'],
                save_gap_data=settings.get('save_gap_data'))
    elif task == 'FEAT':
        save_features(settings['save_file'], data_dir, calc,
                      settings['version'], settings['gg_kwargs'],
                      save_gap_data=settings.get('save_gap_data'))


if __name__ == '__main__':
    call_gpaw()
