from fireworks import FiretaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict

from ase.build import bulk
from ase.calculators.espresso import Espresso
from ase.constraints import UnitCellFilter
from ase import Atoms
import ase.io
from ase.data import atomic_numbers, ground_state_magnetic_moments
from collections import Counter

import collections
import os
import copy

all_pseudos = pseudos = {
    'F': 'F_ONCV_PBE-1.2.upf',
    'O': 'O_ONCV_PBE-1.2.upf',
    'H': 'H_ONCV_PBE-1.2.upf',
    'Li': 'Li_ONCV_PBE-1.2.upf',
}

etot_calc_settings = {
    'control': {
        'calculation': 'scf',
    },
    'system': {},
    'ions': {},
    'cell': {},
    'electrons': {
        'mixing_mode': 'plain',
        'diagonalization': 'david'
    }
}
settings_cats = list(etot_calc_settings.keys())

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def settings_update(d, u):
    d = copy.deepcopy(d)
    for k in settings_cats:
        if (d.get(k) is not None) and (u.get(k) is not None):
            d[k].update(u[k])
        elif u.get(k) is not None:
            d[k] = u[k]
    return d

def run_ase_calc(struct, pseudos, add_settings):
    input_data = copy.deepcopy(etot_calc_settings)
    recursive_update(input_data, add_settings)
    print(input_data)
    calc = Espresso(pseudopotentials=pseudos, input_data=input_data)
    struct.calc = calc
    etot = struct.get_potential_energy()
    fermi_level = calc.get_fermi_level()
    print(etot, fermi_level)
    return etot

#etot_calc_settings.update({'electrons': {'conv_thr': 1e-8}})
recursive_update({'electrons': {'conv_thr': 1e-8}}, etot_calc_settings)
print(etot_calc_settings)

def read_smallmol_xyz(fname, box_size=10):
    atoms = ase.io.read(fname)
    atoms.set_cell([box_size]*3)
    return atoms

def get_atom(el, box_size=10):
    return Atoms(el, cell=[box_size]*3)


def lih_test():
    fname = os.path.join(os.environ.get('ACCDB'), 'Geometries', 'DiPol_LiH.xyz')
    struct = read_smallmol_xyz(fname)
    pseudos = {
        'Li': 'Li_ONCV_PBE-1.2.upf',
        'H': 'H_ONCV_PBE-1.2.upf'
    }
    add_settings = {
        'electrons': {
            'conv_thr': 1e-7
        },
        'system': {
            'ecutwfc': 40
        }
    }
    e_lih = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('H')
    #pseudos = {'H': 'H_ONCV_PBE-1.2.upf'}
    add_settings = {
        'electrons': {
            'conv_thr': 1e-7
        },
        'system': {
            'ecutwfc': 40,
            'nspin': 2,
            'tot_magnetization': 1
        }
    }
    e_h = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('Li')
    #pseudos = {'Li': 'Li_ONCV_PBE-1.2.upf'}
    e_li = run_ase_calc(struct, pseudos, add_settings)
    print('AE', e_h+e_li-e_lih)
    print()


    add_settings = {
        'control': {
            'cider_param_dir': '/home/kyle/Research/qe-test/h/',
            'cider_param_file': 'params_cider',
        },
        'electrons': {
            'conv_thr': 1e-7,
        },
        'system': {
            'ecutwfc': 40,
            'use_cider': True,
            'input_dft': 'SCAN'
        }
    }
    struct = read_smallmol_xyz(fname)
    pseudos = {
        'Li': 'Li_ONCV_PBE-1.2.upf',
        'H': 'H_ONCV_PBE-1.2.upf'
    }
    e_lih = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('H')
    pseudos = {'H': 'H_ONCV_PBE-1.2.upf'}
    add_settings.update({'system': {
        'ecutwfc': 40,
        'nspin': 2,
        'tot_magnetization': 1,
        'use_cider': True,
        'input_dft': 'SCAN'
    }})
    e_h = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('Li')
    pseudos = {'Li': 'Li_ONCV_PBE-1.2.upf'}
    e_li = run_ase_calc(struct, pseudos, add_settings)
    print('AE', e_h+e_li-e_lih)


def hf_test(encut=40):
    fname = os.path.join(os.environ.get('ACCDB'), 'Geometries', 'DiPol_HF.xyz')
    struct = read_smallmol_xyz(fname)
    pseudos = {
        'F': 'F_ONCV_PBE-1.2.upf',
        'H': 'H_ONCV_PBE-1.2.upf'
    }
    add_settings = {
        'control': {
            'pseudo_dir': '/home/kyle/sg15_pseudo'
        },
        'electrons': {
            'conv_thr': 1e-7
        },
        'system': {
            'ecutwfc': encut
        }
    }
    e_lih = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('H')
    add_settings = {
        'control': {
            'pseudo_dir': '/home/kyle/sg15_pseudo'
        },
        'electrons': {
            'conv_thr': 1e-7
        },
        'system': {
            'ecutwfc': encut,
            'nspin': 2,
            'tot_magnetization': 1
        }
    }
    e_h = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('F')
    e_li = run_ase_calc(struct, pseudos, add_settings)
    print('AE', e_h+e_li-e_lih)
    print()


    add_settings = {
        'control': {
            'cider_param_dir': '/home/kyle/Research/qe-test/',
            'cider_param_file': 'tight_params',
            'pseudo_dir': '/home/kyle/sg15_pseudo',
        },
        'electrons': {
            'conv_thr': 1e-7,
        },
        'system': {
            'ecutwfc': encut,
            'use_cider': True,
            'input_dft': 'SCAN'
        }
    }
    struct = read_smallmol_xyz(fname)
    e_lih = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('H')
    add_settings.update({'system': {
        'ecutwfc': encut,
        'nspin': 2,
        'tot_magnetization': 1,
        'use_cider': True,
        'input_dft': 'SCAN'
    }})
    e_h = run_ase_calc(struct, pseudos, add_settings)
    struct = get_atom('F')
    e_li = run_ase_calc(struct, pseudos, add_settings)
    print('AE', e_h+e_li-e_lih)

def ae_test(mol_name, pseudo_dir, encut=40, restricted=True, extra_settings=None):
    fname = os.path.join(os.environ.get('ACCDB'), 'Geometries', '{}.xyz'.format(mol_name))
    struct = read_smallmol_xyz(fname)
    els = struct.get_chemical_symbols()
    atoms = [atomic_numbers[a[0]] for a in els]
    formula = Counter(atoms)

    pseudos = {}
    for el in set(els):
        pseudos[el] = all_pseudos[el]

    add_settings = {
        'control': {
            'pseudo_dir': pseudo_dir
        },
        'electrons': {
            'conv_thr': 1e-7
        },
        'system': {
            'ecutwfc': encut,
            'nspin': 1 if restricted else 2
        }
    }
    if extra_settings is not None:
        add_settings = settings_update(add_settings, extra_settings)
    e_mol = run_ase_calc(struct, pseudos, add_settings)
    ae = -e_mol
    for el in set(els):
        struct = get_atom(el)
        Z = atomic_numbers[el]
        mag = ground_state_magnetic_moments[Z]
        add_settings = {
            'control': {
                'pseudo_dir': pseudo_dir
            },
            'electrons': {
                'conv_thr': 1e-7
            },
            'system': {
                'ecutwfc': encut,
                'nspin': 1 if mag==0 else 2,
                'tot_magnetization': mag
            }
        }
        if extra_settings is not None:
            add_settings = settings_update(add_settings, extra_settings)
        e_at = run_ase_calc(struct, pseudos, add_settings)
        ae += formula[Z] * e_at

    print('AE', ae)

if __name__ == '__main__':
    #hf_test(80)
    cider_settings = {
        'control': {
            'cider_param_dir': '/home/kyle/Research/qe-test/',
            'cider_param_file': 'tight_params',
        },
        'system': {
            'input_dft': 'SCAN',
            'use_cider': True
        }
    }
    #ae_test('DiPol_HF', '/home/kyle/sg15_pseudo', encut=80)
    #print()
    #ae_test('DiPol_H2O', '/home/kyle/sg15_pseudo', encut=80)
    print()
    ae_test('DiPol_H2O', '/home/kyle/sg15_pseudo', encut=80,
            extra_settings=cider_settings)
    print()
    ae_test('DiPol_HF', '/home/kyle/sg15_pseudo', encut=80,
            extra_settings=cider_settings)
