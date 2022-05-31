from fireworks import FiretaskBase, FWAction, Firework
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict

from orchard.workflow_utils import get_save_dir

import time, yaml, os, subprocess, shutil, sys, shlex
import ase.io

import copy

GPAW_CALL_SCRIPT = __file__.replace('gpaw_tasks', 'gpaw_caller')

DEFAULT_GPAW_CALC_SETTINGS = {
    'h': 0.15,
    'xc': 'PBE',
    'txt': 'calc.txt',
    'maxiter': 200,
    'verbose': True,
    'spinpol': False,
    'kpts': (1,1,1),
    'hund': False,
}
DEFAULT_GPAW_CONTROL_SETTINGS = {
    'save_calc' : False,
    'mode' : 1000.0,
    'cider' : None,
}


def setup_gpaw_cmd(struct_path, settings_inp, nproc=None, cmd=None, update_only=False):
    if nproc is None:
        nproc = 1
    if cmd is not None:
        pass
    elif nproc == 1:
        cmd = 'python {call_script} {settings_path} {struct_path}'
    else:
        cmd = 'mpirun -np {nproc} python {call_script} {settings_path} {struct_path}'

    if update_only:
        settings = {'calc': {}, 'control': {'save_calc': False}}
    else:
        settings = {
            'calc': copy.deepcopy(DEFAULT_GPAW_CALC_SETTINGS),
            'control': copy.deepcopy(DEFAULT_GPAW_CONTROL_SETTINGS),
        }
    if 'calc' in settings_inp.keys():
        settings['calc'].update(settings_inp['calc'])
    if 'control' in settings_inp.keys():
        settings['control'].update(settings_inp['control'])
    settings_path = os.path.abspath('./gpaw_settings_tmp.yaml')
    if settings['control']['save_calc']:
        settings['control']['save_calc'] = os.path.abspath('./gpaw_output_tmp.gpw')
    else:
        settings['control']['save_calc'] = None
    with open(settings_path, 'w') as f:
        yaml.dump(settings, f)
    cmd = cmd.format(
        nproc=nproc,
        call_script=GPAW_CALL_SCRIPT,
        settings_path=settings_path,
        struct_path=struct_path,
    )
    return cmd, settings['control']['save_calc'], settings

@explicit_serialize
class GPAWSinglePointSCF(FiretaskBase):

    required_params = ['struct', 'settings', 'method_name', 'system_id']
    optional_params = ['require_converged', 'method_description', 'nproc', 'cmd']

    def run_task(self, fw_spec):
        if self.get('require_converged') is None:
            self['require_converged'] = True
        struct_path = os.path.abspath('./gpaw_fw_tmp.cif')
        ase.io.write(struct_path, self['struct'])
        cmd, save_file, settings = setup_gpaw_cmd(
            struct_path,
            self['settings'],
            nproc=self.get('nproc'),
            cmd=self.get('cmd'),
            update_only=False,
        )

        start_time = time.monotonic()
        proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        if proc.returncode != 0:
            print(proc.stderr.read().decode())
            raise RuntimeError('GPAW Calculation Failed')
        stop_time = time.monotonic()

        with open('gpaw_outdata.tmp', 'r') as f:
            results = yaml.load(f, Loader=yaml.Loader)
        if (not results['converged']) and require_converged:
            raise RuntimeError('GPAW calculation did not converge!')

        update_spec = {
            'e_tot': results['e_tot'],
            'converged': results['converged'],
            'logfile' : results['logfile'],
            'method_name': self['method_name'],
            'method_description': self.get('method_description'),
            'save_file' : save_file,
            'settings' : settings,
            'struct': self['struct'],
            'system_id' : self['system_id'],
            'wall_time' : stop_time - start_time,
        }
        return FWAction(update_spec=update_spec)


@explicit_serialize
class GPAWSinglePointRestart(FiretaskBase):

    required_params = ['new_settings', 'new_method_name', 'restart_file', 'system_id']
    optional_params = ['require_converged', 'new_method_description', 'nproc', 'cmd']

    def run_task(self, fw_spec):
        if self.get('require_converged') is None:
            self['require_converged'] = True
        cmd, save_file, settings = setup_gpaw_cmd(
            self['restart_file'],
            self['new_settings'],
            nproc=self.get('nproc'),
            cmd=self.get('cmd'),
            update_only=True,
        )

        start_time = time.monotonic()
        proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        if proc.returncode != 0:
            print(proc.stderr.read().decode())
            raise RuntimeError('GPAW Calculation Failed')
        stop_time = time.monotonic()

        with open('gpaw_outdata.tmp', 'r') as f:
            results = yaml.load(f, Loader=yaml.Loader)
        if (not results['converged']) and require_converged:
            raise RuntimeError('GPAW calculation did not converge!')

        update_spec = {
            'e_tot': results['e_tot'],
            'converged': results['converged'],
            'logfile' : results['logfile'],
            'method_name': self['new_method_name'],
            'method_description': self.get('new_method_description'),
            'save_file' : save_file,
            'settings' : settings,
            'struct': None,
            'system_id' : self['system_id'],
            'wall_time' : stop_time - start_time,
        }
        return FWAction(update_spec=update_spec)


@explicit_serialize
class SaveGPAWResults(FiretaskBase):

    required_params = ['save_root_dir']
    optional_params = ['no_overwrite']

    def run_task(self, fw_spec):
        save_dir = get_save_dir(
            self['save_root_dir'],
            'PW-KS',
            '',
            fw_spec['system_id'],
            functional=fw_spec['method_name'],
        )
        if self.get('no_overwrite'):
            exist_ok = False
        else:
            exist_ok = True
        os.makedirs(save_dir, exist_ok=exist_ok)

        out_data = {
            'struct': fw_spec['struct'],
            'settings': fw_spec['settings'],
            'e_tot': fw_spec['e_tot'],
            'converged': fw_spec['converged'],
            'wall_time': fw_spec['wall_time'],
            'method_description': fw_spec['method_description'],
        }
        out_file = os.path.join(save_dir, 'run_info.yaml')
        with open(out_file, 'w') as f:
            yaml.dump(out_data, f)

        if fw_spec['logfile'] is not None:
            shutil.copyfile(fw_spec['logfile'], os.path.join(save_dir, 'log.txt'))
        if fw_spec['save_file'] is not None:
            shutil.copyfile(fw_spec['save_file'], os.path.join(save_dir, 'calc.gpw'))

        return FWAction(stored_data={'save_dir': save_dir})


def make_etot_firework(
            struct, settings, method_name, system_id,
            save_root_dir, no_overwrite=False,
            require_converged=True, method_description=None,
            nproc=None, cmd=None,
            name=None,
        ):
    t1 = GPAWSinglePointSCF(
        struct=struct, settings=settings, method_name=method_name, system_id=system_id,
        require_converged=require_converged, method_description=method_description,
        nproc=nproc, cmd=cmd,
    )
    t2 = SaveGPAWResults(save_root_dir=save_root_dir, no_overwrite=no_overwrite)
    return Firework([t1, t2], name=name)


def make_etot_firework_restart(new_settings, new_method_name, system_id, old_method_name,
                               save_root_dir, no_overwrite=False,
                               require_converged=True, new_method_description=None,
                               nproc=None, cmd=None, name=None):
    restart_file = os.path.join(get_save_dir(
        save_root_dir,
        'PW-KS',
        '',
        system_id,
        functional=old_method_name,
    ), 'calc.gpw')
    t1 = GPAWSinglePointRestart(new_settings=new_settings, new_method_name=new_method_name,
                                restart_file=restart_file, system_id=system_id,
                                require_converged=require_converged,
                                new_method_description=new_method_description,
                                nproc=nproc, cmd=cmd)
    t2 = SaveGPAWResults(save_root_dir=save_root_dir, no_overwrite=no_overwrite)
    return Firework([t1, t2], name=name)
