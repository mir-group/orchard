from orchard.vcml_data.eval import values as VCMLValues
from orchard.workflow_utils import VCML_ROOT, ACCDB_ROOT, MLDFTDB_ROOT
import os
import yaml
from ase import Atoms
import ase
from collections import Counter
from ase.data import chemical_symbols, ground_state_magnetic_moments

refvals = VCMLValues()
pbevals = VCMLValues()
refvals.get_values('REF')
pbevals.get_values('PBE')
print(refvals.ecoh)
print(refvals.ecohSys)

method_name = 'PBE'
GEOM_FUNCTIONAL = 'PBE'
VOLUME_NUMBER = 0
DATASET = 'solids'
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
vol = 'volume_{}'.format(VOLUME_NUMBER)
with open(fname, 'r') as f:
    lines = f.readlines()
    struct_ids = [l.strip() for l in lines]
    system_ids = [os.path.join(subdb_name, l.strip(), vol) for l in lines]

def get_sol62_results(sysid):
    data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method_name, sysid)
    with open(os.path.join(data_dir, 'run_info.yaml'), 'r') as f:
        outdata = yaml.load(f, Loader=yaml.Loader)
    atoms = Atoms.fromdict(outdata['struct'])
    formula = Counter(atoms.get_atomic_numbers())
    print(data_dir, formula)
    ntot = 0
    ecoh = outdata['e_tot']
    for Z, count in formula.items():
        ntot += count
        atom_id = 'SOL62/{}/atoms/{}_magmom{}'.format(
            GEOM_FUNCTIONAL,
            chemical_symbols[Z],
            int(ground_state_magnetic_moments[Z]),
        )
        data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method_name, atom_id)
        print('   ', data_dir, os.path.exists(data_dir))
        loaddir = os.path.join(data_dir, 'run_info.yaml')
        if os.path.exists(loaddir):
            with open(loaddir, 'r') as f:
                atom_outdata = yaml.load(f, Loader=yaml.Loader)
            ecoh -= count * atom_outdata['e_tot']
        else:
            print('Missing atom gs', chemical_symbols[Z], ground_state_magnetic_moments[Z])
            return 0
    return ecoh / ntot

ecoh_dict = {}
ev_per_ha = 27.211399
for struct_id, sysid in zip(struct_ids, system_ids):
    ecoh = get_sol62_results(sysid)
    print(ecoh)
    ecoh_dict[struct_id.split('_')[0]] = ev_per_ha * ecoh

print(refvals.ecohSys)
print(refvals.ecoh)

ntot = 0
errtot = 0
errtot2 = 0
pbetot = 0
pbetot2 = 0
#for form_id, val in zip(pbevals.ecohSys, pbevals.ecoh):
for form_id, val, pbeval in zip(refvals.ecohSys, refvals.ecoh, pbevals.ecoh):
    print(form_id, ecoh_dict[form_id] != 0 and abs(pbeval-ecoh_dict[form_id]) > 0.03, val, pbeval, ecoh_dict[form_id])
    if ecoh_dict[form_id] != 0:
        ntot += 1
        err = abs(val - ecoh_dict[form_id])
        perr = abs(val - pbeval)
        errtot += err
        pbetot += perr
        errtot2 += err**2
        pbetot2 += perr**2
        

print(errtot/ntot, pbetot/ntot, (errtot2/ntot)**0.5, (pbetot2/ntot)**0.5)

