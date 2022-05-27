from fireworks import FiretaskBase, FWAction, Firework
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.utilities.fw_serializers import recursive_dict

from orchard import gpaw_caller
from orchard.workflow_utils import get_save_dir

from gpaw import KohnShamConvergenceError

import time, yaml, os, subprocess
import ase.io

GPAW_CALL_SCRIPT = gpaw_caller.__file__

@explicit_serialize
class GPAWSinglePointSCF(FiretaskBase):

    required_params = ['struct', 'settings', 'method_name', 'system_id']
    optional_params = ['require_converged', 'method_description', 'nproc', 'cmd']

    def run_task(self, fw_spec):
        if self.get('require_converged') is None:
            self['require_converged'] = True
        if self.get('nproc') is None:
            nproc = 1
        else:
            nproc = self['nproc']
        if self.get('cmd') is not None:
            cmd = self['cmd']
        elif nproc == 1:
            cmd = 'python {call_script} {settings_path} {struct_path}'
        else:
            cmd = 'mpirun -np {nproc} python {call_script} {settings_path} {struct_path}'
        struct_path = os.path.abspath('./gpaw_fw_tmp.cif')
        settings_path = os.path.abspath('./gpaw_settings_tmp.yaml')
        ase.io.write(struct_path, self['struct'])
        with open(settings_path, 'w') as f:
            yaml.dump(self['settings'], f)
        cmd = cmd.format(
            nproc=nproc,
            call_script=GPAW_CALL_SCRIPT,
            settings_path=settings_path,
            struct_path=struct_path,
        )

        start_time = time.monotonic()
        subprocess.call(cmd, shell=True)
        stop_time = time.monotonic()

        with open('gpaw_outdata.tmp', 'r') as f:
            results = yaml.load(f, Loader=yaml.Loader)
        if (not results['converged']) and require_converged:
            raise RuntimeError('GPAW calculation did not converge!')

        update_spec = {
            'e_tot': results['e_tot'],
            'converged': results['converged'],
            'method_name': self['method_name'],
            'method_description': self.get('method_description'),
            'settings' : self['settings'],
            'struct': self['struct'],
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

