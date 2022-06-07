import os, subprocess
import yaml

fnames = [
    'TODO_RXN1.txt',
    'TODO_RXN2.txt',
    'TODO_BARRIER.txt',
    'TODO_NCI.txt',
    'TODO_INTRA.txt',
]

subdb_names = []
for fname in fnames:
    with open(os.path.join('data_files', fname), 'r') as f:
        subdb_names += [l.strip() for l in f.readlines()]

all_dat = {}
all_list = []
for subdb in subdb_names:
    with open('data_files/GMTKN55/{}.list'.format(subdb), 'r') as f:
        all_list += [l.strip() for l in f.readlines()]
    with open('data_files/GMTKN55/EVAL_{}.yaml'.format(subdb), 'r') as f:
        all_dat.update(yaml.load(f, Loader=yaml.Loader))

with open('data_files/GMTKN55/EVAL_ALL.yaml', 'w') as f:
    yaml.dump(all_dat, f)
print(len(all_dat))
print(len(set(all_list)))

