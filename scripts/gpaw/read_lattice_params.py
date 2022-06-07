import pandas as pd
from orchard.workflow_utils import VCML_ROOT, ACCDB_ROOT, MLDFTDB_ROOT
import os
import shutil
import ase.io.vasp

df = pd.read_csv('data_files/VCML_lattice_params_all.csv', index_col='Solid', header=0)
print(df.loc['Li','Expt'])
print(df['Expt'])

GEOM_FUNCTIONAL = 'PBE'
ddir = 'SOL62'
VOLUME_NUMBER = 0

fname = 'data_files/sol62_solids.txt'

subdb_name = 'SOL62/{}/{}'.format(GEOM_FUNCTIONAL, ddir)
expt_name = 'SOL62/EXPT/{}'.format(ddir)
root_dir = os.path.join(VCML_ROOT, 'datasets/inout_vasp/')
vol = 'volume_{}'.format(VOLUME_NUMBER)
with open(fname, 'r') as f:
    lines = f.readlines()
    struct_ids = [l.strip() for l in lines]
    system_ids = [os.path.join(subdb_name, l.strip(), vol) for l in lines]
    expt_ids = [os.path.join(expt_name, l.strip(), vol) for l in lines]

for struct_id, sysid, exid in zip(struct_ids, system_ids, expt_ids):
    print(struct_id)
    src = os.path.join(root_dir, sysid)
    dest = os.path.join(root_dir, exid)
    os.makedirs(dest, exist_ok=True)
    for fname in ['POSCAR', 'INCAR', 'KPOINTS']:
        shutil.copyfile(os.path.join(src, fname), os.path.join(dest, fname))
    atoms = ase.io.vasp.read_vasp(os.path.join(root_dir, exid, 'POSCAR'))
    formula, celltype = struct_id.split('_')
    a_expt = df.loc[formula, 'Expt']
    if celltype == 'bcc':
        V_expt = a_expt**3 / 2
    else:
        assert celltype in ['fcc', 'diamond', 'zincblende', 'rocksalt']
        V_expt = a_expt**3 / 4
    V_dft = atoms.get_volume()
    ratio = (V_expt / V_dft)**(1.0/3)
    assert abs(ratio-1) < 0.03
    print(atoms.positions)
    atoms.set_cell(atoms.cell*ratio, scale_atoms=True)
    print(atoms.positions)
    assert abs(atoms.get_volume()-V_expt) < 1e-8
    atoms = ase.io.vasp.write_vasp(os.path.join(root_dir, exid, 'POSCAR'), atoms, direct=True)

