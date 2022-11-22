import os
import yaml
from ase import Atoms

config_file = os.path.expanduser('~/.orchard_config.yaml')
if os.path.exists(config_file):
    with open(config_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)
    MLDFTDB_ROOT = settings.get('MLDFTDB_ROOT')
    ACCDB_ROOT = settings.get('ACCDB_ROOT')
    VCML_ROOT = settings.get('VCML_ROOT')
    RXN_ROOT = settings.get('RXN_ROOT')
else:
    MLDFTDB_ROOT = None
    ACCDB_ROOT = None
    VCML_ROOT = None
    RXN_ROOT = None
SAVE_ROOT = MLDFTDB_ROOT

def get_functional_db_name(functional):
    functional = functional.replace(',', '_')
    functional = functional.replace(' ', '_')
    functional = functional.upper()
    return functional

def get_save_dir(root, calc_type, basis, mol_id, functional):
    if functional is not None:
        calc_type = calc_type + '/' + get_functional_db_name(functional)
    return os.path.join(root, calc_type, basis, mol_id)

def load_mol_ids(mol_id_file):
    if not mol_id_file.endswith('.yaml'):
        mol_id_file += '.yaml'
    with open(mol_id_file, 'r') as f:
        contents = yaml.load(f, Loader=yaml.Loader)
    if contents.get('prefix') is not None:
        contents['mols'] = [os.path.join(contents['prefix'], sysid) \
                            for sysid in contents['mols']]
    return contents['mols']

def load_rxns(rxn_list_id, rxndir=None):
    if rxndir is None:
        rxndir = RXN_ROOT
    if rxndir is None:
        raise ValueError('Must provide rxndir or set RXN_ROOT in config')
    rxnfile = os.path.join(rxndir, rxn_list_id) + '.yaml'
    with open(rxnfile, 'r') as f:
        contents = yaml.load(f, Loader=yaml.Loader)
    if contents.get('prefix') is not None:
        prefix = contents.pop('prefix')
        for k, v in contents.items():
            contents[k]['structs'] = [os.path.join(prefix, sysid) \
                                      for sysid in contents[k]['structs']]
    return contents
        

def read_accdb_structure(struct_id):
    fname = '{}.xyz'.format(os.path.join(ACCDB_ROOT, 'Geometries', struct_id))
    with open(fname, 'r') as f:
        #print(fname)
        lines = f.readlines()
        natom = int(lines[0])
        charge_and_spin = lines[1].split()
        charge = int(charge_and_spin[0].strip().strip(','))
        spin = int(charge_and_spin[1].strip().strip(',')) - 1
        symbols = []
        coords = []
        for i in range(natom):
            line = lines[2+i]
            symbol, x, y, z = line.split()
            if symbol.isdigit():
                symbol = int(symbol)
            else:
                symbol = symbol[0].upper() + symbol[1:].lower()
            symbols.append(symbol)
            coords.append([x,y,z])
        struct = Atoms(symbols, positions = coords)
        #print(charge, spin, struct)
    return struct, os.path.join('ACCDB', struct_id), spin, charge
