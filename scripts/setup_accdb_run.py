from orchard.pyscf_tasks import make_etot_firework
from orchard.workflow_utils import MLDFTDB_ROOT, ACCDB_ROOT, read_accdb_structure
import os, sys
import copy

functional = sys.argv[1]
subdb = sys.argv[2]

EXTRA_SETTINGS = {
    'control' : {
        'mol_format': 'ase',
        'spinpol': True,
        'density_fit': True,
        'dftd3': False,
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

method_name = functional
if EXTRA_SETTINGS['control']['dftd3']:
	method_name += '-D3'

dbroot = os.path.join(ACCDB_ROOT, 'Databases/GMTKN/GMTKN55/')
dblist = os.path.join(dbroot, 'GMTKN_{}.list'.format(subdb))
with open(dblist, 'r') as f:
	names = [name.strip() for name in f.readlines()]
struct_dat = [read_accdb_structure(name) for name in names]

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

print(dir(fw_lst[0]))
fwa = fw_lst[0].tasks[0].run_task({})
fwa = fw_lst[0].tasks[1].run_task(fwa.update_spec)
exit()

launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)
