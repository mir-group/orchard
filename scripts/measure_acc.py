from orchard.data import get_accdb_errors
import yaml
from orchard.workflow_utils import ACCDB_ROOT, MLDFTDB_ROOT
import os, sys

fnames = [
    'TODO_RXN1.txt',
    'TODO_RXN2.txt',
    'TODO_BARRIER.txt',
    'TODO_NCI.txt',
    'TODO_INTRA.txt',
]
functional = sys.argv[1]
if len(sys.argv) > 2:
    comp_functional = sys.argv[2]
else:
    comp_functional = None

subdb_names = []
for fname in fnames:
    with open(os.path.join('data_files', fname), 'r') as f:
        subdb_names += [l.strip() for l in f.readlines()]

subdb_names = ['BH76', 'BH76RC']

def get_subdb_mae(subdb):
    with open('data_files/GMTKN55/EVAL_{}.yaml'.format(subdb), 'r') as f:
        dataset = yaml.load(f, Loader=yaml.Loader)
    data_names = list(dataset.keys())

    output = get_accdb_errors(dataset, functional, 'def2-qzvppd', data_names,
                              comp_functional=comp_functional)
    #kcal_per_ha = 627.509608
    print(subdb, output[1])
    #print(subdb, output[1] * kcal_per_ha)

if __name__ == '__main__':
    for subdb in subdb_names:
        try:
            get_subdb_mae(subdb)
        except FileNotFoundError as e:
            raise e
            print('Skipping {}, not all calculations complete'.format(subdb))

