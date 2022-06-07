from orchard.gpaw_tasks import make_etot_firework, make_etot_firework_restart
from orchard.workflow_utils import MLDFTDB_ROOT, VCML_ROOT
import os, sys
import copy

import numpy as np

from ase import Atoms
from ase.build import bulk
import ase.io.vasp
from ase.calculators.vasp import Vasp
from gpaw import Mixer, Davidson

cider_settings = {
	'fname': '/home/kyle/Research/CiderPressDev/SPLINE_MTIGHT_WIDE.joblib',
	'Nalpha': 16,
	'lambd': 1.85,
	'xmix': 0.25,
	'debug': False,
}
base_settings = {
	'calc': {
        'xc': 'PBE',
    },
	'control': {
        'cider': None,#cider_settings,
		'save_calc': True,
		'mode': 1000
	},
}

alkaline_earth = ['Ca_fcc', 'Sr_fcc', 'Ba_bcc']

method_name = base_settings['calc']['xc']
GEOM_FUNCTIONAL = 'PBE'
VOLUME_NUMBER = 0
#DATASET = 'solids'
DATASET = 'atoms'
if DATASET == 'solids':
    fname = 'sol62_solids.txt'
    ddir = 'SOL62'
elif DATASET == 'atoms':
    fname = 'sol62_atoms.txt'
    ddir = 'atoms'
elif DATASET == 'trval':
    fname = 'sol62_trval.txt'
    ddir = 'SOL62'
elif DATASET == 'test':
    fname = 'sol62_test.txt'
    ddir = 'SOL62'
else:
    raise ValueError('Unrecognized Dataset')
fname = os.path.join('data_files', fname)

subdb_name = 'SOL62/{}/{}'.format(GEOM_FUNCTIONAL, ddir)
root_dir = os.path.join(VCML_ROOT, 'datasets/inout_vasp/')
vol = '' if DATASET == 'atoms' else 'volume_{}'.format(VOLUME_NUMBER)
with open(fname, 'r') as f:
    lines = [l.strip() for l in f.readlines()]
    system_ids = [os.path.join(subdb_name, l, vol) for l in lines]
    if DATASET == 'atoms':
        magmoms = {sysid : int(l[-1]) for sysid, l in zip(system_ids, lines)}

def get_sol62_fw(sysid, magmom=None):
    data_dir = os.path.join(root_dir, sysid)
    all_settings = copy.deepcopy(base_settings)
    settings = all_settings['calc']
    control = all_settings['control']
    if ddir == 'SOL62':
        atoms = ase.io.vasp.read_vasp(os.path.join(data_dir, 'POSCAR'))
        vcalc = Vasp(atoms)
        vcalc.read_incar(os.path.join(data_dir, 'INCAR'))
        vcalc.read_kpoints(os.path.join(data_dir, 'KPOINTS'))
        settings['kpts'] = {'size': vcalc.input_params['kpts'], 'gamma': vcalc.input_params['gamma']}
        ismear = vcalc.int_params['ismear']
        sigma = vcalc.float_params['sigma']
        settings['occupations'] = {'name': 'fermi-dirac', 'width': sigma}
        for am_id in alkaline_earth:
            if am_id in sysid:
                #settings['occupations']['width'] = 0.04
                settings['mixer'] = Mixer(0.02, 5, 100)
                settings['eigensolver'] = 'cg'#Davidson(3)
                break
        settings['convergence'] = {'energy': 1e-5, 'density': 1.0e-4, 'eigenstates': 4.0e-8, 'bands': 'occupied'}
        control['mode'] = vcalc.float_params['encut']
        ispin = vcalc.int_params['ispin']
        if ispin == 1:
            spinpol = False
        elif ispin == 2:
            spinpol = True
        else:
            raise ValueError('Unsupported ispin')
        settings['spinpol'] = spinpol
        magmoms = vcalc.list_float_params['magmom']
        if magmoms is not None:
            control['magmom'] = vcalc.list_float_params['magmom']
        print(type(atoms.get_initial_magnetic_moments()))
    else:
        atoms = ase.io.vasp.read_vasp(os.path.join(data_dir, 'POSCAR'))
        print(data_dir)
        atoms.set_pbc(False)
        settings['kpts'] = {'size': (1,1,1), 'gamma': True}
        settings['occupations'] = {'name': 'fermi-dirac', 'width': 0.00, 'fixmagmom': True}
        #settings['convergence'] = {'energy': 1e-5, 'density': 1.0e-4, 'eigenstates': 1.0e-5, 'bands': 'occupied'}
        settings['convergence'] = {'energy': 1e-5, 'density': 1.0e-2, 'eigenstates': 1.0e-5, 'bands': 'occupied'}
        control['mode'] = 1000 #vcalc.float_params['encut']
        settings['spinpol'] = (magmom > 0)
        settings['mixer'] = Mixer(0.02, 5, 200)
        control['eigensolver'] = {'name': 'dav', 'niter': 4}
        control['magmom'] = [magmom]
    return make_etot_firework(
	    atoms, all_settings, method_name, sysid, MLDFTDB_ROOT, name='{}_{}'.format(sysid, method_name)
    )
fws = []
for i, sysid in enumerate(system_ids):
    fws.append(get_sol62_fw(sysid, magmom=magmoms[sysid] if DATASET=='atoms' else None))

from fireworks import LaunchPad, Firework
launchpad = LaunchPad.auto_load()
for fw in fws:
    launchpad.add_wf(fw)
    #if 'Ba_bcc' in fw.tasks[0]['system_id']:
    #    launchpad.add_wf(fw)
    #if 'Fe_bcc' in fw.tasks[0]['system_id']:
    #    launchpad.add_wf(fw)
    #    fw0 = fw
    #if 'B_' in fw.tasks[0]['system_id']:
    #    fw0 = fw

exit()
fw0 = fw0.to_dict()
fw0 = Firework.from_dict(fw0)
fwa = fw0.tasks[0].run_task({})
fwa = fw0.tasks[1].run_task(fwa.update_spec)
exit()

new_settings = {
	'control' : {
		'cider' : cider_settings,
	}
}
fw = make_etot_firework_restart(
	new_settings, 'CIDER', 'METALS/Zn', 'PBE', MLDFTDB_ROOT, nproc=4,
)

print(dir(fw))
fwa = fw.tasks[0].run_task({})
fwa = fw.tasks[1].run_task(fwa.update_spec)
exit()
