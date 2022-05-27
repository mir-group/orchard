import copy
from pyscf import gto, dft, scf

CALC_TYPES = {
    'RKS' : dft.rks.RKS,
    'UKS' : dft.uks.UKS,
}

'''
All PySCF settings supported:
{
    'control' : {
        'spinpol': bool,
        'density_fit': bool,
        'df_basis': None or str (basis name)
        'radi_method': None or str (func name)
        'remove_linear_dep': bool,
    },
    'mol' : {
        'basis': str, default 'def2-qzvppd'
        'spin': int, default 0
        'charge': int, default 0
        'verbose': int, default 3
    },
    'calc': {
        'xc': str,
        'conv_tol': float,
        ... other calc settings as needed
    },
    'grids': {
        'level': int,
        'prune': None or str,
        'atom_grid': dict
    }
    'cider': None or { # (overrides 'xc' in calc)
        'mlfunc_filename': str,
        'xmix': float,
        'xkernel': str, libxc name of exchange kernel,
        'ckernel': str, libxc name of correlation kernel,
        'debug': bool,
    }
}
'''
DEFAULT_SETTINGS = {
    'control' : {
        'mol_format': 'xyz',
        'spinpol': False,
        'density_fit': False,
        'dftd3': False,
        'df_basis': None,
        'remove_linear_dep': True,
    },
    'mol' : {
        'basis': 'def2-qzvppd',
        'spin': 0,
        'charge': 0,
        'verbose': 3,
    },
    'calc' : {
        'xc': 'PBE',
    },
    'grids': {},
}

def setup_calc(atoms, settings_inp):
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    inp_keys = list(settings_inp.keys())
    for k in list(settings.keys()):
        elif k in inp_keys:
            settings[k].update(settings_inp[k])
    if 'cider' in inp_keys:
        settings['cider'] = settings_inp['cider']

    mol = gto.Mole()
    fmt = settings['control']['mol_format']
    if fmt == 'xyz_file':
        mol.atom = atoms
    elif fmt in ['xyz', 'raw', 'zmat']:
        mol.atom = gto.mole.fromstring(atoms, format=fmt)
    elif fmt == 'pyscf':
        mol.atom = atoms
    elif fmt == 'ase':
        from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
        mol.atom = atoms_from_ase(atoms)
    mol.__dict__.update(settings['mol'])
    mol.build()

    if settings.get('cider') is None:
        calc = dft.UKS(mol) if settings['control']['spinpol'] else dft.RKS(mols)
    else:
        from mldftdat.dft.ri_cider import setup_cider_calc
        calc = setup_cider_calc(
            mol,
            joblib.load(settings['cider']['mlfunc_filename']),
            spinpol=settings['control']['spinpol'],
            xkernel=settings['cider']['xkernel'],
            ckernel=settings['cider']['ckernel'],
            xmix=settings['cider']['xmix'],
            debug=settings['cider']['debug'],
        )
    calc.__dict__.update(settings['calc'])
    if settings['control']['density_fit']:
        calc = calc.density_fit()
        if settings['control'].get('df_basis') is not None:
            calc.with_df.auxbasis = settings['control']['df_basis']
    if settings['control']['remove_linear_dep']:
        calc = calc.apply(scf.addons.remove_linear_dep_)

    calc.grids.__dict__.update(settings['grids'])
    if settings['control'].get('dftd3'):
        from pyscf import dftd3
        calc = dftd3.dftd3(calc)

    return calc

def update_calc_settings(calc, settings_update):
    calc.__dict__.update(settings_update)
    return calc

