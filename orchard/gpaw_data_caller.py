import yaml
import os
import sys
import numpy as np
from gpaw import restart, PW
from ase.parallel import paropen
from pyscf.lib import chkfile
from ase.units import Ha


def get_exx(data_dir, calc, encut, kpts, p_be=None):
    """

    :param save_dir:
    :param calc:
    :param p_be: (p_vbm, p_cbm), (s, k, n) for each
    :return:
    """
    from gpaw.hybrids.energy import non_self_consistent_energy
    calc.set(mode=PW(encut))
    if kpts is not None:
        calc.set(kpts = kpts)
    calc.get_potential_energy()
    eterms = non_self_consistent_energy(calc, 'EXX')
    data = {}
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


def save_features(save_file, data_dir, calc, version, gg_kwargs, p_be=None):
    from ciderpress.gpaw.analysis import get_features
    res = get_features(calc, p_i=p_be, version=version, **gg_kwargs)
    rho_res = get_features(calc, p_i=p_be, version='l')
    with open(os.path.join(data_dir, 'exx_data.yaml'), 'r') as f:
        data = yaml.load(f, Loader=yaml.CLoader)
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
        'coord': None,
        'wt': all_wt,
        'nspin': nspin,
    })
    data['val'] = data['exx'] * all_wt / (nspin * all_wt.sum())
    data['val'] = np.stack([data['val'], data['val']]) # sums to exx
    if calc.world.rank == 0:
        chkfile.dump(save_file, 'train_data', data)


def call_gpaw():
    with open(sys.argv[1], 'r') as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    data_dir = settings['data_dir']
    task = settings['task'] # should be EXX or FEAT
    atoms, calc = restart(os.path.join(data_dir, 'calc.gpw'))

    if settings.get('save_gap_data'):
        from ase.dft.bandgap import bandgap
        gap, p_vbm, p_cbm = bandgap(calc)
        p_be = (p_vbm, p_cbm)
    else:
        p_be = None

    if task == 'EXX':
        get_exx(data_dir, calc, settings['encut'], settings['kpts'], p_be=p_be)
    elif task == 'FEAT':
        save_features(settings['save_file'], data_dir, calc,
                      settings['version'], settings['gg_kwargs'],
                      p_be=p_be)


if __name__ == '__main__':
    call_gpaw()
