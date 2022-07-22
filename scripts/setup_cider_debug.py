from orchard.pyscf_tasks import make_etot_firework, make_etot_firework_restart
from orchard.workflow_utils import MLDFTDB_ROOT, ACCDB_ROOT, read_accdb_structure
import os, sys
import copy
import yaml

cider_dir = sys.argv[1]
subdb = sys.argv[2]

EXTRA_SETTINGS = {
    'control' : {
        'dftd3': False,
        'dftd4': False,
        'only_dfj': True,
    },
    'calc' : {
        'xc': 'PBE',
    },
}

cider_joblib = os.path.join(cider_dir, 'model.joblib')
cider_desc = os.path.join(cider_dir, 'description.yaml')
with open(cider_desc, 'r') as f:
    cider_desc = yaml.load(f, Loader=yaml.Loader)

CIDER_SETTINGS = { # (overrides 'xc' in calc)
    'mlfunc_filename': cider_joblib,
    'xmix': 0.25,
    'xkernel': 'GGA_X_PBE',
    'ckernel': 'GGA_C_PBE',
    'debug': False,
}

if CIDER_SETTINGS['debug']:
    functional = os.path.join('DEBUG_CIDER', cider_desc['name'])
else:
    functional = os.path.join('TEST_CIDER', cider_desc['name'])
EXTRA_SETTINGS['cider'] = CIDER_SETTINGS

method_name = functional
if EXTRA_SETTINGS['control']['dftd3']:
    method_name += '-D3'
if EXTRA_SETTINGS['control']['dftd4']:
    assert EXTRA_SETTINGS['control']['dftd3'] == False, 'No D3 and D4 as the same time'
    method_name += '-D4'
    EXTRA_SETTINGS['control']['dftd4_functional'] = 'PBE0'

dbroot = os.path.join(ACCDB_ROOT, 'Databases/GMTKN/GMTKN55/')
dblist = os.path.join(dbroot, 'GMTKN_{}.list'.format(subdb))
with open(dblist, 'r') as f:
    names = [name.strip() for name in f.readlines()]
struct_dat = [read_accdb_structure(name) for name in names]

maxz = 0
spinpol = False
for struct, mol_id, spin, charge in struct_dat:
    maxz = max(maxz, np.max(struct.get_atomic_numbers()))
    if spin != 0:
        spinpol = True

EXTRA_SETTINGS['control']['spinpol'] = spinpol
maxz = min(maxz, 36)
EXTRA_SETTINGS['cider']['amax'] = (maxz/6)**2 * 1000

fw_lst = []
for struct, mol_id, spin, charge in struct_dat:
    mol_id = mol_id.replace('ACCDB', 'GMTKN55')
    settings = copy.deepcopy(EXTRA_SETTINGS)
    new_fw = make_etot_firework_restart(
        settings, method_name, mol_id, 'def2-qzvppd', 'PBE',
        MLDFTDB_ROOT, no_overwrite=False,
        require_converged=True, new_method_description=cider_desc,
        name=method_name + '_' + mol_id
    )
    fw_lst.append(new_fw)
'''
print(dir(fw_lst[0]))
for fw in fw_lst:
    if fw.tasks[0]['system_id'] == 'GMTKN55/BH76_hoch3fcomp2':
        break
else:
    exit()
spec = {}
for task_num in range(len(fw.tasks)):
    fwa = fw.tasks[task_num].run_task(spec)
    spec.update(fwa.update_spec)
exit()
'''
from fireworks import LaunchPad
launchpad = LaunchPad.auto_load()
for fw in fw_lst:
    launchpad.add_wf(fw)

