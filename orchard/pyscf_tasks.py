from fireworks import FiretaskBase, FWAction, Firework
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict

from orchard import pyscf_caller
from orchard.workflow_utils import get_save_dir

from pyscf import lib
from ase import Atoms

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
        'ecp': 'def2-qzvppd',
        'spin': 0,
        'charge': 0,
        'verbose': 3,
    },
    'calc' : {
        'xc': 'PBE',
    },
    'grids': {},
}

def get_pyscf_settings(settings_inp, default_settings=DEFAULT_PYSCF_SETTINGS):
    settings = copy.deepcopy(default_settings)
    inp_keys = list(settings_inp.keys())
    for k in list(settings.keys()):
        if k in inp_keys:
            settings[k].update(settings_inp[k])
    for optk in ['cider', 'jax']:
        if optk in inp_keys:
            settings[optk] = settings_inp[optk]
    return settings


@explicit_serialize
class SCFCalc(FiretaskBase):

    required_params = ['struct', 'settings', 'method_name', 'system_id']
    optional_params = ['require_converged', 'method_description']

    def run_task(self, fw_spec):
        settings = get_pyscf_settings(self['settings'])
        start_time = time.monotonic()
        calc = pyscf_caller.setup_calc(Atoms.fromdict(self['struct']), settings)
        calc.kernel()
        stop_time = time.monotonic()
        if self.get('require_converged') is None:
            self['require_converged'] = True
        if (not calc.converged) and self['require_converged']:
            raise RuntimeError("SCF calculation did not converge!")
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
class LoadSCFCalc(FiretaskBase):

    required_params = ['save_root_dir', 'method_name', 'basis', 'system_id']

    def run_task(self, fw_spec):
        load_dir = get_save_dir(
            self['save_root_dir'],
            'KS',
            self['basis'],
            self['system_id'],
            functional=self['method_name'],
        )
        hdf5file = os.path.join(load_dir, 'data.hdf5')
        in_file = os.path.join(load_dir, 'run_info.yaml')
        with open(in_file, 'r') as f:
            in_data = yaml.load(f, Loader=yaml.Loader)
        calc = pyscf_caller.setup_calc(Atoms.fromdict(in_data['struct']), in_data['settings'])
        calc.e_tot = lib.chkfile.load(hdf5file, 'calc/e_tot')
        calc.mo_coeff = lib.chkfile.load(hdf5file, 'calc/mo_coeff')
        calc.mo_energy = lib.chkfile.load(hdf5file, 'calc/mo_energy')
        calc.mo_occ = lib.chkfile.load(hdf5file, 'calc/mo_occ')
        update_spec = {
            'basis' : self['basis'],
            'calc' : calc,
            'method_name' : self['method_name'],
            'settings' : in_data['settings'],
            'struct' : in_data['struct'],
            'system_id' : self['system_id'],
        }
        return FWAction(update_spec=update_spec)


@explicit_serialize
class SCFCalcFromRestart(FiretaskBase):

    required_params = ['new_settings', 'new_method_name']
    optional_params = ['require_converged', 'new_method_description']

    def run_task(self, fw_spec):
        settings = get_pyscf_settings(self['new_settings'], default_settings=fw_spec['settings'])
        start_time = time.monotonic()
        calc = pyscf_caller.setup_calc(Atoms.fromdict(fw_spec['struct']), settings)
        calc.kernel(dm0=fw_spec['calc'].make_rdm1())
        stop_time = time.monotonic()
        if self.get('require_converged') is None:
            self['require_converged'] = True
        if (not calc.converged) and self['require_converged']:
            raise RuntimeError("SCF calculation did not converge!")
        update_spec = {
            'calc' : calc,
            'e_tot': calc.e_tot,
            'converged': calc.converged,
            'method_name': self['new_method_name'],
            'method_description': self.get('new_method_description'),
            'pyscf_atoms' : calc.mol._atom,
            'settings' : settings,
            'wall_time' : stop_time - start_time,
        }
        return FWAction(update_spec=update_spec)


@explicit_serialize
class SaveSCFResults(FiretaskBase):

    required_params = ['save_root_dir']
    optional_params = ['no_overwrite', 'write_data']

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
        if self.get('write_data') is None:
            self['write_data'] = True
        if self['write_data']:
            hdf5file = os.path.join(save_dir, 'data.hdf5')
            lib.chkfile.save(hdf5file, 'calc/e_tot', calc.e_tot)
            lib.chkfile.save(hdf5file, 'calc/mo_coeff', calc.mo_coeff)
            lib.chkfile.save(hdf5file, 'calc/mo_energy', calc.mo_energy)
            lib.chkfile.save(hdf5file, 'calc/mo_occ', calc.mo_occ)
        out_data = {
            'struct': fw_spec['struct'],
            'settings': fw_spec['settings'],
            'e_tot': fw_spec['e_tot'],
            'e_tot_readable': float(fw_spec['e_tot']), # since e_tot is numpy double scalar
            'converged': fw_spec['converged'],
            'conv_tol': calc.conv_tol,
            'wall_time': fw_spec['wall_time'],
            'method_description': fw_spec['method_description'],
        }
        out_file = os.path.join(save_dir, 'run_info.yaml')
        with open(out_file, 'w') as f:
            yaml.dump(out_data, f)

        return FWAction(stored_data={'save_dir': save_dir})


@explicit_serialize
class RunAnalysis(FiretaskBase):

    required_params = ['save_root_dir', 'system_id']
    optional_params = ['grids_level', 'cider_kwargs_and_version']

    def get_cider_features(self, analyzer, restricted):
        from ciderpress.density import get_exchange_descriptors2
        gg_kwargs = self['cider_kwargs_and_version']
        version = gg_kwargs.pop('version')
        descriptor_data = get_exchange_descriptors2(
            analyzer, restricted=restricted, version=version,
            **gg_kwargs
        )
        analyzer.set('cider_descriptor_data', descriptor_data)

    def run_task(self, fw_spec):
        from ciderpress.analyzers import ElectronAnalyzer
        calc = fw_spec['calc']
        analyzer = ElectronAnalyzer.from_calc(calc, self.get('grids_level'))
        analyzer.perform_full_analysis()
        save_dir = get_save_dir(
            self['save_root_dir'], 'KS',
            calc.mol.basis, self['system_id'],
            fw_spec['method_name']
        )
        save_file = os.path.join(save_dir,
            'analysis_L{}.hdf5'.format(analyzer.grids_level))
        if self.get('cider_kwargs_and_version') is not None:
            self.get_cider_features(analyzer, analyzer.dm.ndim==2)
        analyzer.dump(save_file)

        return FWAction(stored_data={'save_dir': save_dir})


def make_etot_firework(
            struct, settings, method_name, system_id,
            save_root_dir, no_overwrite=False,
            require_converged=True, method_description=None,
            write_data=None, name=None):
    struct = struct.todict()
    t1 = SCFCalc(
            struct=struct, settings=settings, method_name=method_name, system_id=system_id,
            require_converged=require_converged, method_description=method_description
        )
    t2 = SaveSCFResults(save_root_dir=save_root_dir, no_overwrite=no_overwrite,
                        write_data=write_data)
    return Firework([t1, t2], name=name)


def make_etot_firework_restart(new_settings, new_method_name, system_id,
                               old_basis, old_method_name,
                               save_root_dir, no_overwrite=False,
                               require_converged=True, new_method_description=None,
                               write_data=None, name=None):
    t1 = LoadSCFCalc(
        save_root_dir=save_root_dir, method_name=old_method_name,
        basis=old_basis, system_id=system_id
    )
    t2 = SCFCalcFromRestart(new_settings=new_settings, new_method_name=new_method_name,
                            require_converged=require_converged,
                            new_method_description=new_method_description)
    t3 = SaveSCFResults(save_root_dir=save_root_dir, no_overwrite=no_overwrite,
                        write_data=write_data)
    return Firework([t1, t2, t3], name=name)

def make_analysis_firework(method_name, system_id, basis, save_root_dir,
                           grids_level=None, name=None, **kwargs):
    t1 = LoadSCFCalc(
        save_root_dir=save_root_dir, method_name=method_name,
        basis=basis, system_id=system_id,
    )
    tasks = [t1]
    if grids_level is None or isinstance(grids_level, int):
        tasks.append(RunAnalysis(
            save_root_dir=save_root_dir,
            system_id=system_id,
            grids_level=grids_level, **kwargs
        ))
    elif isinstance(grids_level, (tuple, list)):
        for lvl in grids_level:
            tasks.append(RunAnalysis(
                save_root_dir=save_root_dir,
                system_id=system_id,
                grids_level=lvl, **kwargs
            ))
    else:
        raise ValueError('Unsupported grids_level')
    return Firework(tasks, name=name)

