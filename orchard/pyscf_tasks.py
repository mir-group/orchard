from fireworks import FiretaskBase, FWAction, Firework
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict

from orchard import pyscf_caller
from orchard.workflow_utils import get_save_dir

from pyscf import lib

import time
import yaml
import os
import copy


DEFAULT_PYSCF_SETTINGS = {
    'control' : {
        'mol_format': 'xyz',
        'spinpol': False,
        'density_fit': False,
        'dftd3': False,
        'df_basis': None,
        'remove_linear_dep': True,
    },
    'mol' : {
        'basis': 'def2-qzvppd',
        'spin': 0,
        'charge': 0,
        'verbose': 3,
    },
    'calc' : {
        'xc': 'PBE',
    },
    'grids': {},
}

def get_pyscf_settings(settings_inp):
    settings = copy.deepcopy(DEFAULT_PYSCF_SETTINGS)
    inp_keys = list(settings_inp.keys())
    for k in list(settings.keys()):
        if k in inp_keys:
            settings[k].update(settings_inp[k])
    if 'cider' in inp_keys:
        settings['cider'] = settings_inp['cider']
    return settings


@explicit_serialize
class SCFCalc(FiretaskBase):

    required_params = ['struct', 'settings', 'method_name', 'system_id']
    optional_params = ['require_converged', 'method_description']

    def run_task(self, fw_spec):
        settings = get_pyscf_settings(self['settings'])
        start_time = time.monotonic()
        calc = pyscf_caller.setup_calc(self['struct'], settings)
        calc.kernel()
        stop_time = time.monotonic()
        if self.get('require_converged') is None:
            self['require_converged'] = True
        if (not calc.converged) and self['require_converged']:
            assert RuntimeError("SCF calculation did not converge!")
        update_spec = {
            'calc' : calc,
            'e_tot': calc.e_tot,
            'converged': calc.converged,
            'method_name': self['method_name'],
            'method_description': self.get('method_description'),
            'pyscf_atoms' : calc.mol._atom,
            'settings' : settings,
            'struct': self['struct'],
            'system_id' : self['system_id'],
            'wall_time' : stop_time - start_time,
        }
        return FWAction(update_spec=update_spec)


@explicit_serialize
class SaveSCFResults(FiretaskBase):

    required_params = ['save_root_dir']
    optional_params = ['no_overwrite']

    def run_task(self, fw_spec):
        save_dir = get_save_dir(
            self['save_root_dir'],
            'KS',
            fw_spec['calc'].mol.basis,
            fw_spec['system_id'],
            functional=fw_spec['method_name']
        )
        if self.get('no_overwrite'):
            exist_ok = False
        else:
            exist_ok = True
        os.makedirs(save_dir, exist_ok=exist_ok)

        calc = fw_spec['calc']
        chkmol = os.path.join(save_dir, 'mol.chk')
        lib.chkfile.save_mol(calc.mol, chkmol)
        hdf5file = os.path.join(save_dir, 'data.hdf5')
        lib.chkfile.save(hdf5file, 'calc/e_tot', calc.e_tot)
        lib.chkfile.save(hdf5file, 'calc/mo_coeff', calc.mo_coeff)
        lib.chkfile.save(hdf5file, 'calc/mo_energy', calc.mo_energy)
        lib.chkfile.save(hdf5file, 'calc/mo_occ', calc.mo_occ)
        out_data = {
            'struct': fw_spec['struct'],
            'settings': fw_spec['settings'],
            'e_tot': fw_spec['e_tot'],
            'converged': fw_spec['converged'],
            'conv_tol': calc.conv_tol,
            'wall_time': fw_spec['wall_time'],
            'method_description': fw_spec['method_description'],
        }
        out_file = os.path.join(save_dir, 'run_info.yaml')
        with open(out_file, 'w') as f:
            yaml.dump(out_data, f)

        return FWAction(stored_data={'save_dir': save_dir})


def make_etot_firework(
            struct, settings, method_name, system_id,
            save_root_dir, no_overwrite=False,
            require_converged=True, method_description=None,
            name=None,
        ):
    t1 = SCFCalc(struct=struct, settings=settings, method_name=method_name, system_id=system_id,
                 require_converged=require_converged, method_description=method_description)
    t2 = SaveSCFResults(save_root_dir=save_root_dir, no_overwrite=no_overwrite)
    return Firework([t1, t2], name=name)

