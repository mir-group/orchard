from orchard.gpaw_tasks import make_etot_firework
from orchard.workflow_utils import MLDFTDB_ROOT, ACCDB_ROOT, read_accdb_structure
import os, sys
import copy

from ase import Atoms
from ase.build import bulk

cider_settings = {
	'fname': '/home/kyle/Research/CiderPressDev/SPLINE_MTIGHT_WIDE.joblib',
	'Nalpha': 16,
	'lambd': 1.85,
	'xmix': 0.25,
	'debug': False,
}
settings = {
	'kpts': (12, 12, 12),
	'cider': cider_settings,
}

#atoms = Atoms('Ne')
#atoms.center(vacuum=3)
#print(atoms.pbc)
atoms = bulk('Zn')

fw = make_etot_firework(
	atoms, settings, 'CIDER', 'ATOMS/Ne', MLDFTDB_ROOT
)
print(dir(fw))
fwa = fw.tasks[0].run_task({})
fwa = fw.tasks[1].run_task(fwa.update_spec)
exit()
