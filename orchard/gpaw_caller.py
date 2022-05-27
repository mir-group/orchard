from gpaw import GPAW, PW
import copy

DEFAULT_SETTINGS = {
    'h': 0.15,
    'xc': 'PBE',
    'mode': 1000.0,
    'txt': 'calc.txt',
    'maxiter': 200,
    'verbose': True,
    'spinpol': False,
    'kpts': (1,1,1),
    'hund': False,
}

def setup_gpaw(settings_inp):
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    settings.update(settings_inp)
    if settings.get('cider') is not None:
        from mldftdat.gpaw.cider_paw import CiderGGAPASDW
        cider_settings = settings.pop('cider')
        fname = cider_settings.pop('fname')
        settings['xc'] = CiderGGAPASDW.from_joblib(
            fname, **cider_settings
        )
    if settings['mode'] != 'fd':
        settings['mode'] = PW(settings['mode'])
    return GPAW(**settings)

def call_gpaw():
    import yaml
    import sys
    import ase.io
    from ase.parallel import paropen
    from ase.units import Ha
    from gpaw import KohnShamConvergenceError

    with open(sys.argv[1], 'r') as f:
        settings = yaml.load(f, Loader=yaml.Loader)
    atoms = ase.io.read(sys.argv[2])
    atoms.calc = setup_gpaw(settings)
    try:
        e_tot = atoms.get_potential_energy()
        converged = True
    except KohnShamConvergenceError as e:
        converged = False

    with paropen('gpaw_outdata.tmp', 'w') as f:
        f.write('e_tot : {}\n'.format(e_tot / Ha))
        f.write('converged : {}\n'.format(converged))


if __name__ == '__main__':
    call_gpaw()
