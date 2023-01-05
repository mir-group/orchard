from gpaw import GPAW, PW
from gpaw import Davidson, CG, RMMDIIS
from ase import Atoms
from ase.units import Ha, Bohr
import copy, os

def setup_gpaw(settings_inp, calc=None):
    settings = settings_inp['calc']
    control = settings_inp['control']
    if control.get('cider') is not None:
        from mldftdat.gpaw.cider_paw import CiderGGAPASDW, CiderMGGAPASDW
        cider_settings = control['cider']
        fname = cider_settings.pop('fname')
        try:
            settings['xc'] = CiderGGAPASDW.from_joblib(
                fname, **cider_settings
            )
        except ValueError:
            settings['xc'] = CiderMGGAPASDW.from_joblib(
                fname, **cider_settings
            )

    if control.get('multipole_corr') is not None:
        from gpaw.poisson import PoissonSolver
        from gpaw.poisson_moment import MomentCorrectionPoissonSolver
        mom = 1 + control['multipole_corr']
        mom = mom * mom
        settings['poissonsolver'] = MomentCorrectionPoissonSolver(poissonsolver=PoissonSolver(),
                                                                  moment_corrections=mom)
    if control.get('eigensolver') is not None:
        eigd = control.get('eigensolver')
        eigname = eigd.pop('name')
        if eigname == 'dav':
            solver = Davidson(**eigd)
        elif eigname == 'rmm-diis':
            solver = RMMDIIS(**eigd)
        elif eigname == 'cg':
            solver = CG(**eigd)
        else:
            raise ValueError('Unrecognized solver name')
        settings['eigensolver'] = solver
    if control.get('mode') is None:
        if calc is None:
            raise ValueError('Need mode or calc')
    elif control['mode'] == 'lcao':
        settings['mode'] = 'lcao'
        settings['eigensolver'] = None
    elif control['mode'] != 'fd':
        settings['mode'] = PW(control['mode']) # mode = encut
        if settings['calc'].get('h') is None:
            # Default h should fit encut
            encut = control['mode']
            gcut = np.sqrt(2 * encut / Ha)
            settings['calc']['h'] = (Bohr * np.pi) / (2 * gcut)
    else:
        settings['mode'] = 'fd'

    if calc is None:
        calc = GPAW(**settings)
    else:
        calc.set(**settings)
    
    if settings.get('txt') is None:
        settings_inp['calc']['txt'] = 'calc.txt'
        calc.set(txt=settings_inp['calc']['txt'])

    if control.get('parallel') is not None:
        calc.parallel.update(control['parallel'])

    return calc

def get_nscf_routine(settings_inp):
    settings = settings_inp['calc']
    control = settings_inp['control']
    if control.get('cider') is not None:
        from mldftdat.gpaw.cider_paw import CiderGGAPASDW, CiderMGGAPASDW
        cider_settings = control['cider']
        fname = cider_settings.pop('fname')
        try:
            settings['xc'] = CiderGGAPASDW.from_joblib(
                fname, **cider_settings
            )
        except ValueError:
            cider_settings['debug'] = False # Not going to use potential anyway
                                            # debug not implemented for MGGA
            settings['xc'] = CiderMGGAPASDW.from_joblib(
                fname, **cider_settings
            )
        def routine(atoms):
            return get_nscf_energy_nonhybrid(atoms, settings['xc'])
    elif settings.get('xc') in ['EXX', 'PBE0', 'HSE03', 'HSE06', 'B3LYP']:
        def routine(atoms):
            return get_nscf_energy_hybrid(atoms, settings['xc'], settings['kpts'])
    else:
        def routine(atoms):
            return get_nscf_energy_nonhybrid(atoms, settings['xc'])
    return routine

def get_total_energy(atoms):
    return atoms.get_potential_energy()

def get_nscf_energy_hybrid(atoms, xcname, hybrid_kpts):
    from gpaw.hybrids.energy import non_self_consistent_energy
    #atoms.calc.reset()
    #atoms.calc.initialize()
    e0 = atoms.get_potential_energy()
    #atoms.calc.set(kpts=(8,8,8))
    atoms.calc.set(kpts=hybrid_kpts, txt='-', verbose=1)
    e0t = atoms.get_potential_energy()
    if xcname == 'EXX':
        et = non_self_consistent_energy(atoms.calc, xcname=xcname)[3:].sum()
        return et
    else:
        et = non_self_consistent_energy(atoms.calc, xcname=xcname).sum()
        return et - e0t + e0

def get_nscf_energy_nonhybrid(atoms, xc):
    e0 = atoms.calc.get_potential_energy()
    return e0 + atoms.calc.get_xc_difference(xc)

def call_gpaw():
    import yaml
    import sys
    import ase.io
    from ase.parallel import paropen
    from ase.units import Ha
    from gpaw import KohnShamConvergenceError

    with open(sys.argv[1], 'r') as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    restart_file = settings.get('restart_file')
    if restart_file is not None:
        from gpaw import restart
        atoms, calc = restart(restart_file)
        if settings['control'].get('nscf'):
            routine = get_nscf_routine(settings)
        else:
            routine = get_total_energy
            setup_gpaw(settings, calc=calc)
    else:
        atoms = Atoms.fromdict(settings.pop('struct'))
        atoms.calc = setup_gpaw(settings)
        magmoms = settings['control'].get('magmom')
        if magmoms is not None:
            atoms.set_initial_magnetic_moments(magmoms)
        routine = get_total_energy

    try:
        e_tot = routine(atoms)
        converged = True
    except KohnShamConvergenceError as e:
        e_tot = float("NaN")
        converged = False

    with paropen('gpaw_outdata.tmp', 'w') as f:
        f.write('e_tot : {}\n'.format(e_tot / Ha))
        f.write('converged : {}\n'.format(converged))
        #txtfile = settings['calc'].get('txt')
        #if txtfile is not None:
        #    assert os.path.exists(txtfile)
        #    f.write('logfile : {}\n'.format(os.path.abspath(txtfile)))
        #else:
        #    f.write('logfile : None\n')

    if settings['control'].get('save_calc') is not None:
        assert settings['control']['save_calc'].endswith('.gpw')
        atoms.calc.write(settings['control']['save_calc'], mode='all')


if __name__ == '__main__':
    call_gpaw()
