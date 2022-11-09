from pyscf import gto, dft, scf
from copy import deepcopy

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
        'mol_format': str
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

def setup_calc(atoms, settings):
    settings = deepcopy(settings)
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
        calc = dft.UKS(mol) if settings['control']['spinpol'] else dft.RKS(mol)
    else:
        from mldftdat.dft.ri_cider import setup_cider_calc
        import joblib
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
    
    if settings['control'].get('sgx_params') is not None:
        sgx_params = settings['control'].get('sgx_params')
        from pyscf import sgx
        auxbasis = settings['control'].get('df_basis') or 'def2-universal-jfit'
        pjs = sgx_params.pop('pjs')
        calc = sgx.sgx_fit(calc, auxbasis=auxbasis, pjs=pjs)
        calc.with_df.__dict__.update(**sgx_params)
    elif settings['control']['density_fit']:
        calc = calc.density_fit(only_dfj=settings['control'].get('only_dfj') or False)
        if settings['control'].get('df_basis') is not None:
            calc.with_df.auxbasis = settings['control']['df_basis']

    if settings['calc'].get('nlc') is not None:
        calc.nlcgrids.level = 1

    if settings['control']['remove_linear_dep']:
        calc = calc.apply(scf.addons.remove_linear_dep_)

    calc.grids.__dict__.update(settings['grids'])
    if settings['control'].get('dftd3'):
        from pyscf import dftd3
        calc = dftd3.dftd3(calc)
        d3v = settings['control'].get('dftd3_version')
        if d3v is not None:
            print('DFTD3: Setting version to', d3v)
            calc.with_dftd3.version = d3v
    elif settings['control'].get('dftd4'):
        import dftd4.pyscf as pyd4
        calc = pyd4.energy(calc)
        d4func = settings['control'].get('dftd4_functional')
        if d4func is not None:
            calc.with_dftd4 = pyd4.DFTD4Dispersion(
                calc.mol, xc=d4func.upper().replace(" ", "")
            )

    if settings['control'].get('soscf'):
        calc = calc.newton()

    return calc

def update_calc_settings(calc, settings_update):
    calc.__dict__.update(settings_update)
    return calc

