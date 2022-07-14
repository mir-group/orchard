from orchard.vcml_data.eval import values as VCMLValues
from orchard.workflow_utils import VCML_ROOT, ACCDB_ROOT, MLDFTDB_ROOT
import os, sys
import yaml
from ase import Atoms
import ase
from collections import Counter
from ase.data import chemical_symbols, ground_state_magnetic_moments

refvals = VCMLValues()
pbevals = VCMLValues()
refvals.get_values('REF')
#refvals.get_values('HSE06')
pbevals.get_values('PBE')
#print(refvals.ecoh)
#print(refvals.ecohSys)

method_name = sys.argv[1]#'DEBUG_CIDER/GGA_520_002'
if len(sys.argv) > 2:
    comp_method = sys.argv[2]
else:
    comp_method = None

GEOM_FUNCTIONAL = 'PBE'
VOLUME_NUMBER = 0
#DATASET = 'solids'
DATASET = 'trval'
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

def get_sol62_results(sysid, method):
    data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method, sysid)
    with open(os.path.join(data_dir, 'run_info.yaml'), 'r') as f:
        outdata = yaml.load(f, Loader=yaml.Loader)
    atoms = Atoms.fromdict(outdata['struct'])
    formula = Counter(atoms.get_atomic_numbers())
    #print(data_dir, formula)
    ntot = 0
    ecoh = outdata['e_tot']
    for Z, count in formula.items():
        ntot += count
        e_atom_min = 1e10
        for magmom in range(0,10):
            #if magmom != int(ground_state_magnetic_moments[Z]):
            #    continue
            atom_id = 'SOL62/{}/atoms/{}_magmom{}'.format(
                GEOM_FUNCTIONAL,
                chemical_symbols[Z],
                magmom,#int(ground_state_magnetic_moments[Z]),
            )
            data_dir = os.path.join(MLDFTDB_ROOT, 'PW-KS', method, atom_id)
            #print('   ', data_dir, os.path.exists(data_dir))
            loaddir = os.path.join(data_dir, 'run_info.yaml')
            if os.path.exists(loaddir):
                with open(loaddir, 'r') as f:
                    atom_outdata = yaml.load(f, Loader=yaml.Loader)
                e_atom_min = min(e_atom_min, atom_outdata['e_tot'])
                #ecoh -= count * atom_outdata['e_tot']
            else:
                continue
                #print('Missing atom gs', chemical_symbols[Z], ground_state_magnetic_moments[Z])
                #return 0
        ecoh -= count * e_atom_min
    return ecoh / ntot

ecoh_dict = {}
if comp_method is not None:
    ref_dict = {}
ev_per_ha = 27.211399
print('NUM_IDS', len(struct_ids))
for struct_id, sysid in zip(struct_ids, system_ids):
    #if 'Si_' not in sysid:
    #    continue
    ecoh = get_sol62_results(sysid, method_name)
    #print(ecoh)
    ecoh_dict[struct_id.split('_')[0]] = ev_per_ha * ecoh
    if comp_method is not None:
        ecoh = get_sol62_results(sysid, comp_method)
        ref_dict[struct_id.split('_')[0]] = ev_per_ha * ecoh

#print(ecoh_dict)
#print(refvals.ecohSys)
#print(refvals.ecoh)

ntot = 0
errtot = 0
errtot2 = 0
pbetot = 0
pbetot2 = 0
#for form_id, val in zip(pbevals.ecohSys, pbevals.ecoh):
count = 0
ndat = 0
diff_thr = 0.1 # 0.04336
for form_id, val, pbeval in zip(refvals.ecohSys, refvals.ecoh, pbevals.ecoh):
    if form_id not in ecoh_dict:
        continue
    if comp_method is not None:
        val = ref_dict[form_id]
    print(form_id, ecoh_dict[form_id] != 0 and abs(pbeval-ecoh_dict[form_id]) > diff_thr, val, pbeval-val, '       \t', ecoh_dict[form_id]-val)
    ndat += 1
    if abs(pbeval-ecoh_dict[form_id]) > diff_thr:
        count += 1
    if ecoh_dict[form_id] != 0:
        ntot += 1
        err = abs(val - ecoh_dict[form_id])
        perr = abs(val - pbeval)
        errtot += err
        pbetot += perr
        errtot2 += err**2
        pbetot2 += perr**2
        
print(count, ndat, errtot/ntot, pbetot/ntot, (errtot2/ntot)**0.5, (pbetot2/ntot)**0.5)

