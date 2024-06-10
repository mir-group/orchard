import os
from orchard.workflow_utils import ACCDB_ROOT, MLDFTDB_ROOT, read_accdb_structure, get_save_dir
import yaml
import numpy as np

KCAL_PER_HA = 627.509608

def get_run_energy_and_nbond(dirname):
    with open(os.path.join(dirname, 'run_info.yaml'), 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    nb = 0#get_nbond(data['mol']['atom'])
    return data['e_tot'], nb

def get_run_total_energy(dirname):
    with open(os.path.join(dirname, 'run_info.yaml'), 'r') as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data['e_tot']

def get_accdb_data(formula, FUNCTIONAL, BASIS, per_bond=False):
    pred_energy = 0
    if per_bond:
        nbond = 0
        #nbond = None
    for sname, count in zip(formula['structs'], formula['counts']):
        if sname.startswith('atoms/'):
            mol_id = sname
        else:
            struct, mol_id, spin, charge = read_accdb_structure(sname)
            mol_id = mol_id.replace('ACCDB', 'GMTKN55')
        CALC_TYPE = 'KS'
        dname = get_save_dir(MLDFTDB_ROOT, CALC_TYPE, BASIS, mol_id, FUNCTIONAL)
        if per_bond:
            en, nb = get_run_energy_and_nbond(dname)
            pred_energy += count * en
            nbond += count * nb
        else:
            pred_energy += count * get_run_total_energy(dname)

    if per_bond:
        return pred_energy, formula['energy'], abs(nbond)
    else:
        return pred_energy, formula['energy']

def get_accdb_pts(snames, FUNCTIONAL, BASIS):
    vals = {}
    for sname in snames:
        dname = get_save_dir(MLDFTDB_ROOT, 'KS', BASIS, 'GMTKN55/'+sname, FUNCTIONAL)
        vals[sname] = get_run_total_energy(dname)
    return vals

def get_accdb_rvals(formulas, vals):
    rvals = {}
    for rname, formula in formulas.items():
        pred_energy = 0
        for sname, count in zip(formula['structs'], formula['counts']):
            pred_energy += count * vals[sname]
        rvals[rname] = pred_energy
    return rvals

def get_accdb_formulas(dataset_eval_name):
    with open(dataset_eval_name, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.split(',')
        formulas = {}
        for line in lines:
            counts = line[1:-1:2]
            structs = line[2:-1:2]
            energy = float(line[-1])
            counts = [int(c) for c in counts]
            formulas[line[0]] = {'structs': structs, 'counts': counts, 'energy': energy}
    return formulas

def parse_dataset_eval(subdb_names, dataset_fname):
    if dataset_fname.endswith('.yaml'):
        with open(dataset_fname, 'r') as f:
            formulas = yaml.load(f, Loader=yaml.Loader)
    else:
        formulas = get_accdb_formulas(dataset_fname)
    cats = {}
    sumabs = {}
    counts = {}
    for name in subdb_names:
        cats[name] = []
        sumabs[name] = 0
        counts[name] = 0
    for dname, formula in list(formulas.items()):
        for name in subdb_names:
            if dname.startswith(name):
                cats[name].append(dname)
                counts[name] += 1
                sumabs[name] += abs(formula['energy'])
                break
        else:
            raise RuntimeError('Datapoint {} not matched to subdb'.format(dname))
    return cats, counts, sumabs

def get_accdb_performance(dataset_eval_name, FUNCTIONAL, BASIS, data_names,
                          per_bond=False, comp_functional=None):
    formulas = get_accdb_formulas(dataset_eval_name)
    result = {}
    errs = []
    nbonds = 0
    for data_point_name, formula in list(formulas.items()):
        if data_point_name not in data_names:
            continue
        pred_energy, energy, nbond = get_accdb_data(formula, FUNCTIONAL, BASIS,
                                                    per_bond=True)
        nbonds += nbond
        result[data_point_name] = {
            'pred' : pred_energy,
            'true' : energy
        }
        print(data_point_name, pred_energy * KCAL_PER_HA, energy * KCAL_PER_HA)
        if comp_functional is not None:
            pred_ref, _, _ = get_accdb_data(formula, comp_functional, BASIS,
                                            per_bond=True)
            energy = pred_ref
            result[data_point_name]['true'] = pred_ref
        #print(pred_energy-energy, pred_energy, energy)
        errs.append(pred_energy-energy)
    errs = np.array(errs)
    #print(errs.shape)
    me = np.mean(errs)
    mae = np.mean(np.abs(errs))
    rmse = np.sqrt(np.mean(errs**2))
    std = np.std(errs)
    if per_bond:
        return nbonds, np.sum(errs) / nbonds, np.sum(np.abs(errs)) / nbonds
    else:
        return me, mae, rmse, std, result

def get_accdb_errors(formulas, FUNCTIONAL, BASIS, data_names,
                     comp_functional=None):
    errs = []
    refs = []
    result = {}
    for data_name in data_names:
        #print(data_name)
        pred_energy, energy = get_accdb_data(formulas[data_name], FUNCTIONAL, BASIS)
        exact_ref = energy
        if comp_functional is not None:
            energy, _ = get_accdb_data(formulas[data_name], comp_functional, BASIS)
            energy *= KCAL_PER_HA
        pred_energy *= KCAL_PER_HA
        result[data_name] = {
            'pred' : pred_energy,
            'true' : energy,
            'weight': 1.0 / (formulas[data_name].get('noise_factor') or 1.0),
        }
        #print(data_name, pred_energy-energy)
        errs.append(pred_energy-energy)
        #refs.append(energy)
        refs.append(exact_ref)
    errs = np.array(errs)
    me = np.mean(errs)
    mae = np.mean(np.abs(errs))
    rmse = np.sqrt(np.mean(errs**2))
    std = np.std(errs)
    return me, mae, rmse, std, result, len(errs), np.mean(np.abs(refs))

def get_subdb_mae(subdb, xc, comp_xc=None, data_names=None,
                  return_count=False, ae=False, return_result=False):
    eval_file = 'GMTKN55/EVAL_{}.yaml'.format(subdb)
    if ae:
        eval_file = eval_file.replace('GMTKN55/EVAL_', 'GMTKN55/AE2_EVAL_')
    eval_file = os.path.join(ACCDB_ROOT, 'Databases/GMTKN', eval_file)
    with open(eval_file, 'r') as f:
        dataset = yaml.load(f, Loader=yaml.Loader)
    if data_names is None:
        data_names = list(dataset.keys())
    elif isinstance(data_names, str):
        with open(data_names, 'r') as f:
            data_names = list(yaml.load(f, Loader=yaml.Loader).keys())
            data_names = [d.split('/')[-1] for d in data_names]
    data_names = [d for d in data_names if d != 'prefix']

    output = get_accdb_errors(dataset, xc, 'def2-qzvppd', data_names,
                              comp_functional=comp_xc)
    #print(subdb, output[0], output[1])
    print(subdb, output[1])
    #print(output[1])
    if return_count:
        if return_result:
            return output[-2], output[-1], output[1], output[4]
        else:
            return output[-2], output[-1], output[1]
    if return_result:
        return output[1], output[4]
    return output[1]

def get_band_gap(hdf5file):
    from pyscf import lib
    print(hdf5file)
    mo_energy = lib.chkfile.load(hdf5file, 'calc/mo_energy')
    mo_occ = lib.chkfile.load(hdf5file, 'calc/mo_occ')
    moe = np.asarray(mo_energy)
    moo = np.asarray(mo_occ)
    homo = np.max(moe[moo > 1e-10])
    lumo = np.min(moe[moo < 1e-10])
    return lumo - homo

def get_subdb_gaps(subdb, xc, basis):
    eval_file = 'GMTKN55/{}.list'.format(subdb)
    eval_file = os.path.join(ACCDB_ROOT, 'Databases/GMTKN', eval_file)
    with open(eval_file, 'r') as f:
        mol_ids = ['GMTKN55/{}'.format(l.strip()) for l in f.readlines()]
    gaps = {}
    for mol_id in mol_ids:
        dname = get_save_dir(MLDFTDB_ROOT, 'KS', basis, mol_id, xc)
        dname = os.path.join(dname, 'data.hdf5')
        gaps[mol_id] = get_band_gap(dname)
    return gaps

def get_weighted_loss(subdb, xc, comp_xc=None, data_names=None, return_count=False, ae=False):
    eval_file = 'GMTKN55/EVAL_{}.yaml'.format(subdb)
    if ae:
        eval_file = eval_file.replace('GMTKN55/EVAL_', 'GMTKN55/AE2_EVAL_')
    eval_file = os.path.join(ACCDB_ROOT, 'Databases/GMTKN', eval_file)
    with open(eval_file, 'r') as f:
        dataset = yaml.load(f, Loader=yaml.Loader)
    if data_names is None:
        data_names = list(dataset.keys())
    elif isinstance(data_names, str):
        with open(data_names, 'r') as f:
            data_names = list(yaml.load(f, Loader=yaml.Loader).keys())
            data_names = [d.split('/')[-1] for d in data_names]
    data_names = [d for d in data_names if d != 'prefix']

    output = get_accdb_errors(dataset, xc, 'def2-qzvppd', data_names,
                              comp_functional=comp_xc)

    res = output[4]
    loss = 0
    for _, r in res.items():
        loss += ( (r['pred'] - r['true']) * r['weight'] )**2
    loss = np.sqrt(loss / output[-2])
    if return_count:
        return output[-2], output[-1], loss
    print(subdb, loss)
    return loss

