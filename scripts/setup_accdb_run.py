from orchard.pyscf_tasks import make_etot_firework, make_etot_firework_restart
from orchard.workflow_utils import MLDFTDB_ROOT, ACCDB_ROOT, read_accdb_structure
import os, sys
import copy
import numpy as np

functional = sys.argv[1]
subdb = sys.argv[2]

EXTRA_SETTINGS = {
    'control' : {
        'mol_format': 'ase',
        'spinpol': True,
        'density_fit': True,
        'only_dfj': True,
        'dftd3': False,
        'dftd3_version': 4,
        'dftd4': True,
        'dftd4_functional': None,
        'df_basis': 'def2-universal-jkfit',
        'remove_linear_dep': True,
    },
    'mol' : {
        'basis': 'def2-qzvppd',
        'spin': 0,
        'charge': 0,
        'verbose': 4,
    },
    'calc' : {
        'xc': functional,
    },
}
CIDER_SETTINGS = { # (overrides 'xc' in calc)
    #'mlfunc_filename': '/home/kyle/Research/CiderPressDev/SPLINE_MTIGHT_WIDE.joblib',
    'mlfunc_filename': '/n/home01/kbystrom/repos/CiderPressDev/SPLINE_MTIGHT_WIDE.joblib',
    'xmix': 0.25,
    'xkernel': 'GGA_X_PBE',
    'ckernel': 'GGA_C_PBE',
    'debug': False,
}
SGX_SETTINGS = {
    'pjs' : False,
    'direct_scf_tol' : 1e-13,
    'dfj' : True,
    'grids_level_i' : 1,
    'grids_level_f' : 1,
}
#EXTRA_SETTINGS['control']['sgx_params'] = SGX_SETTINGS

# set CIDER
if functional == 'CIDER':
    EXTRA_SETTINGS['cider'] = CIDER_SETTINGS
    EXTRA_SETTINGS['calc']['xc'] = 'PBE'
    EXTRA_SETTINGS['control']['dftd4_functional'] = 'PBE0'
else:
    from pyscf.dft import libxc
    if (np.array(libxc.hybrid_coeff(functional)) > 0).any() \
            or (np.array(libxc.rsh_coeff(functional)) > 0).any():
        EXTRA_SETTINGS['control']['sgx_params'] = SGX_SETTINGS
        print('SGX')

method_name = functional
if EXTRA_SETTINGS['control']['dftd3']:
    method_name += '-D3'
if EXTRA_SETTINGS['control']['dftd4']:
    assert EXTRA_SETTINGS['control']['dftd3'] == False, 'No D3 and D4 as the same time'
    method_name += '-D4'

dbroot = os.path.join(ACCDB_ROOT, 'Databases/GMTKN/GMTKN55/')
dblist = os.path.join(dbroot, 'GMTKN_{}.list'.format(subdb))
with open(dblist, 'r') as f:
    names = [name.strip() for name in f.readlines()]
struct_dat = [read_accdb_structure(name) for name in names]

for struct, mol_id, spin, charge in struct_dat:
    if spin != 0:
        spinpol = True
        break
else:
    spinpol = False

EXTRA_SETTINGS['control']['spinpol'] = spinpol

fw_lst = []
for struct, mol_id, spin, charge in struct_dat:
    mol_id = mol_id.replace('ACCDB', 'GMTKN55')
    settings = copy.deepcopy(EXTRA_SETTINGS)
    settings['mol']['spin'] = spin
    settings['mol']['charge'] = charge
    fw_lst.append(make_etot_firework(
        struct, settings, method_name, mol_id,
        MLDFTDB_ROOT, name='{}_{}'.format(method_name, mol_id)
    ))

#print(dir(fw_lst[0]))
#fw = fw_lst[0]
#spec = {}
#for task_num in range(len(fw.tasks)):
#    fwa = fw.tasks[task_num].run_task(spec)
#    spec.update(fwa.update_spec)
#exit()

from fireworks import LaunchPad
launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)
