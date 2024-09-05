#!/usr/bin/env python
# orchard: Utilities to training and analyzing machine learning-based density functionals
# Copyright (C) 2024 The President and Fellows of Harvard College
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# Author: Kyle Bystrom <kylebystrom@gmail.com>
#

import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import yaml
from ciderpress.analyzers import ElectronAnalyzer, RHFAnalyzer, UHFAnalyzer
from ciderpress.data import (
    get_total_weights_spherical,
    get_unique_coord_indexes_spherical,
)
from ciderpress.density import DESC_VERSION_LIST, GG_AMIN, get_exchange_descriptors
from ciderpress.descriptors import FAST_DESC_VERSION_LIST, get_descriptors
from pyscf.lib import chkfile

from orchard.workflow_utils import SAVE_ROOT, get_save_dir, load_mol_ids

"""
Script to compile a dataset from the CIDER DB for training a CIDER functional.
"""


def compile_dataset_old(
    DATASET_NAME,
    MOL_IDS,
    SAVE_ROOT,
    FUNCTIONAL,
    BASIS,
    spherical_atom=False,
    version="a",
    sparse_level=None,
    analysis_level=1,
    **gg_kwargs
):

    all_descriptor_data = []
    all_rho_data = []
    all_values = []
    all_weights = []
    cutoffs = [0]

    for MOL_ID in MOL_IDS:
        logging.info("Computing descriptors for {}".format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, "KS", BASIS, MOL_ID, FUNCTIONAL)
        start = time.monotonic()
        analyzer = ElectronAnalyzer.load(
            data_dir + "/analysis_L{}.hdf5".format(analysis_level)
        )
        if sparse_level is not None:
            Analyzer = UHFAnalyzer if analyzer.atype == "UHF" else RHFAnalyzer
            analyzer = Analyzer(analyzer.mol, analyzer.dm, grids_level=sparse_level)
            analyzer.perform_full_analysis()
            level = sparse_level
        else:
            analyzer.get_rho_data()
            level = analysis_level
        if isinstance(level, int):
            sparse_tag = "_{}".format(level)
        else:
            sparse_tag = "_{}_{}".format(level[0], level[1])
        restricted = False if analyzer.atype == "UHF" else True
        end = time.monotonic()
        logging.info("Analyzer load time {}".format(end - start))

        if spherical_atom:
            start = time.monotonic()
            indexes = get_unique_coord_indexes_spherical(analyzer.grids.coords)
            uwts = get_total_weights_spherical(
                analyzer.grids.coords[indexes],
                analyzer.grids.coords,
                analyzer.grids.weights,
            )
            end = time.monotonic()
            logging.info("Index scanning time {}".format(end - start))
        start = time.monotonic()
        if restricted:
            descriptor_data = get_exchange_descriptors(
                analyzer, restricted=True, version=version, **gg_kwargs
            )
        else:
            descriptor_data_u, descriptor_data_d = get_exchange_descriptors(
                analyzer, restricted=False, version=version, **gg_kwargs
            )
            descriptor_data = np.append(descriptor_data_u, descriptor_data_d, axis=1)
        end = time.monotonic()
        logging.info("Get descriptor time {}".format(end - start))
        values = analyzer.get("ex_energy_density")
        rho_data = analyzer.rho_data
        if spherical_atom:
            if not restricted:
                raise ValueError("Spherical atom not supported with spin pol.")
            values = values[indexes]
            descriptor_data = descriptor_data[:, indexes]
            rho_data = rho_data[:, indexes]
            weights = uwts
        else:
            weights = analyzer.grids.weights
        if not restricted:
            values = 2 * np.append(values[0], values[1])
            rho_data = 2 * np.append(rho_data[0], rho_data[1], axis=1)
            weights = 0.5 * np.append(weights, weights)

        all_rho_data.append(rho_data)
        all_values.append(values)
        all_weights.append(weights)
        all_descriptor_data.append(descriptor_data)
        cutoffs.append(cutoffs[-1] + values.size)

    all_rho_data = np.concatenate(all_rho_data, axis=-1)
    all_values = np.concatenate(all_values)
    all_weights = np.concatenate(all_weights)
    all_descriptor_data = np.concatenate(all_descriptor_data, axis=-1)

    DATASET_NAME = os.path.basename(DATASET_NAME)
    save_dir = os.path.join(
        SAVE_ROOT, "DATASETS", FUNCTIONAL, BASIS, version + sparse_tag, DATASET_NAME
    )
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    rho_file = os.path.join(save_dir, "rho.npy")
    desc_file = os.path.join(save_dir, "desc.npy")
    val_file = os.path.join(save_dir, "val.npy")
    wt_file = os.path.join(save_dir, "wt.npy")
    cut_file = os.path.join(save_dir, "cut.npy")
    np.save(rho_file, all_rho_data)
    np.save(desc_file, all_descriptor_data)
    np.save(val_file, all_values)
    np.save(wt_file, all_weights)
    np.save(cut_file, np.array(cutoffs))
    settings = {
        "DATASET_NAME": DATASET_NAME,
        "MOL_IDS": MOL_IDS,
        "SAVE_ROOT": SAVE_ROOT,
        "FUNCTIONAL": FUNCTIONAL,
        "BASIS": BASIS,
        "spherical_atom": spherical_atom,
        "version": version,
    }
    settings.update(gg_kwargs)
    with open(os.path.join(save_dir, "settings.yaml"), "w") as f:
        yaml.dump(settings, f)


def compile_single_system(
    save_file, analyzer_file, version, sparse_level, orbs, save_baselines, gg_kwargs
):
    start = time.monotonic()
    analyzer = ElectronAnalyzer.load(analyzer_file)
    if sparse_level is not None:
        old_analyzer = analyzer
        Analyzer = UHFAnalyzer if analyzer.atype == "UHF" else RHFAnalyzer
        analyzer = Analyzer(
            analyzer.mol,
            analyzer.dm,
            grids_level=sparse_level,
            mo_occ=old_analyzer.mo_occ,
            mo_coeff=old_analyzer.mo_coeff,
            mo_energy=old_analyzer.mo_energy,
        )
        if "e_tot_orig" in old_analyzer._data:
            analyzer._data["xc_orig"] = old_analyzer.get("xc_orig")
            analyzer._data["exc_orig"] = old_analyzer.get("exc_orig")
            analyzer._data["e_tot_orig"] = old_analyzer.get("e_tot_orig")
        analyzer.perform_full_analysis()
    else:
        analyzer.get_rho_data()
    end = time.monotonic()
    logging.info("Analyzer load time {}".format(end - start))

    start = time.monotonic()
    desc = get_descriptors(analyzer, version=version, orbs=orbs, **gg_kwargs)
    rho_data = get_descriptors(analyzer, version="l", orbs=orbs, **gg_kwargs)
    if orbs is not None:
        desc, ddesc, eigvals = desc
        rho_data, drho_data, _ = rho_data
    end = time.monotonic()
    logging.info("Get descriptor time {}".format(end - start))
    values = analyzer.get("ex_energy_density")
    weights = analyzer.grids.weights
    coords = analyzer.grids.coords
    if isinstance(analyzer, UHFAnalyzer):
        spinpol = True
        values = np.stack([values[0], values[1]])
        # NOTE not doing this factor of 2 thing anymore since
        # training loop must be spin-aware to account for
        # correlation functional
        # rho_data = 2 * np.stack(rho_data[0], rho_data[1])
        # weights *= 0.5
    else:
        values = values[np.newaxis, :]
        desc = desc[np.newaxis, :]
        spinpol = False

    data = {
        "rho": rho_data,
        "desc": desc,
        "val": values,
        "coord": coords,
        "wt": weights,
        "nspin": 2 if spinpol else 1,
    }
    if orbs is not None:
        data["dval"] = intk_to_strk(analyzer.calculate_vxc_on_mo("HF", orbs))
        data["ddesc"] = intk_to_strk(ddesc)
        data["eigvals"] = intk_to_strk(eigvals)
        data["drho_data"] = intk_to_strk(drho_data)
    if save_baselines:
        data["xc_orig"] = analyzer.get("xc_orig")
        data["exc_orig"] = analyzer.get("exc_orig")
        data["e_tot_orig"] = analyzer.get("e_tot_orig")
    dirname = os.path.dirname(os.path.abspath(save_file))
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    chkfile.dump(save_file, "train_data", data)


def intk_to_strk(d):
    if not isinstance(d, dict):
        return d
    nd = {}
    for k, v in d.items():
        nd[str(k)] = intk_to_strk(v)
    return nd


def compile_dataset(
    DESC_NAME,
    DATASET_NAME,
    MOL_IDS,
    SAVE_ROOT,
    FUNCTIONAL,
    BASIS,
    version="b",
    sparse_level=None,
    analysis_level=1,
    save_gap_data=False,
    save_baselines=True,
    make_fws=False,
    skip_existing=False,
    save_dir=None,
    **gg_kwargs
):
    if version not in FAST_DESC_VERSION_LIST:
        raise ValueError("Unsupported version for new dataset module")

    if save_gap_data:
        orbs = {"O": [0], "U": [0]}
    else:
        orbs = None

    if sparse_level is None:
        level = analysis_level
    else:
        level = sparse_level
    if isinstance(level, int):
        sparse_tag = "_{}".format(level)
    else:
        sparse_tag = "_{}_{}".format(level[0], level[1])

    if save_dir is None:
        save_dir = os.path.join(
            SAVE_ROOT,
            "DATASETS",
            FUNCTIONAL,
            BASIS,
            version + sparse_tag,
            DESC_NAME,
        )
    else:
        save_dir = os.path.join(save_dir, DESC_NAME)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    settings = {
        "DATASET_NAME": DATASET_NAME,
        "DESC_NAME": DESC_NAME,
        "MOL_IDS": MOL_IDS,
        "SAVE_ROOT": SAVE_ROOT,
        "FUNCTIONAL": FUNCTIONAL,
        "BASIS": BASIS,
        "version": version,
    }
    settings.update(gg_kwargs)
    print(save_dir, SAVE_ROOT, DESC_NAME)
    print(os.path.join(save_dir, "{}_settings.yaml".format(DATASET_NAME)))
    with open(
        os.path.join(save_dir, "{}_settings.yaml".format(DATASET_NAME)), "w"
    ) as f:
        yaml.dump(settings, f)

    if make_fws:
        from orchard.pyscf_tasks import StoreFeatures

        fwlist = {}

    for MOL_ID in MOL_IDS:
        logging.info("Computing descriptors for {}".format(MOL_ID))
        data_dir = get_save_dir(SAVE_ROOT, "KS", BASIS, MOL_ID, FUNCTIONAL)
        save_file = os.path.join(save_dir, MOL_ID + ".hdf5")
        if os.path.exists(save_file) and skip_existing:
            print("Already exists, skipping:", MOL_ID)
            continue
        analyzer_file = data_dir + "/analysis_L{}.hdf5".format(analysis_level)
        args = (
            save_file,
            analyzer_file,
            version,
            sparse_level,
            orbs,
            save_baselines,
            gg_kwargs,
        )
        if make_fws:
            fwname = "feature_{}_{}".format(version, MOL_ID)
            fwlist[fwname] = StoreFeatures(args=args)
        else:
            compile_single_system(*args)

    if make_fws:
        return fwlist


def main():
    logging.basicConfig(level=logging.INFO)

    m_desc = "Compile dataset of XC descriptors"

    parser = ArgumentParser(description=m_desc)
    parser.add_argument(
        "mol_id_file", type=str, help="yaml file from which to read mol_ids to parse"
    )
    parser.add_argument("basis", metavar="basis", type=str, help="basis set code")
    parser.add_argument(
        "--functional",
        metavar="functional",
        type=str,
        default=None,
        help="exchange-correlation functional, HF for Hartree-Fock",
    )
    parser.add_argument(
        "--spherical-atom",
        action="store_true",
        default=False,
        help="whether dataset contains spherical atoms",
    )
    parser.add_argument(
        "--version", default="c", type=str, help="version of descriptor set. Default c"
    )
    parser.add_argument("--gg-a0", default=8.0, type=float)
    parser.add_argument("--gg-facmul", default=1.0, type=float)
    parser.add_argument("--gg-amin", default=GG_AMIN, type=float)
    parser.add_argument(
        "--gg-vvmul",
        default=1.0,
        type=float,
        help="For version b only, mul to get second coord exponent",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        type=str,
        help="customize data directories with this suffix",
    )
    parser.add_argument(
        "--analysis-level",
        default=1,
        type=int,
        help="Level of analysis to search for each system, looks for analysis_L{analysis-level}.hdf5",
    )
    parser.add_argument(
        "--sparse-grid",
        default=None,
        type=int,
        nargs="+",
        help="use a sparse grid to compute features, etc. If set, recomputes data.",
    )
    parser.add_argument("--make-fws", action="store_true")
    parser.add_argument("--save-gap-data", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip system if save_file exists already",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        type=str,
        help="override default save directory for features",
    )
    args = parser.parse_args()

    version = args.version.lower()
    if version not in DESC_VERSION_LIST:
        raise ValueError("Unsupported descriptor set")

    mol_ids = load_mol_ids(args.mol_id_file)
    if args.mol_id_file.endswith(".yaml"):
        mol_id_code = args.mol_id_file[:-5]
    else:
        mol_id_code = args.mol_id_file

    if args.sparse_grid is None:
        sparse_level = None
    elif len(args.sparse_grid) == 1:
        sparse_level = args.sparse_grid[0]
    elif len(args.sparse_grid) == 2:
        sparse_level = (args.sparse_grid[0], args.sparse_grid[1])
    else:
        raise ValueError("Sparse grid must be 1 or 2 integers")

    gg_kwargs = {"amin": args.gg_amin, "a0": args.gg_a0, "fac_mul": args.gg_facmul}
    if version in ["b", "d", "e"]:
        gg_kwargs["vvmul"] = args.gg_vvmul
    res = compile_dataset(
        "_UNNAMED" if args.suffix is None else args.suffix,
        mol_id_code.upper().split("/")[-1],
        mol_ids,
        SAVE_ROOT,
        args.functional,
        args.basis,
        # spherical_atom=args.spherical_atom,
        version=version,
        analysis_level=args.analysis_level,
        sparse_level=sparse_level,
        save_gap_data=args.save_gap_data,
        make_fws=args.make_fws,
        skip_existing=args.skip_existing,
        save_dir=args.save_dir,
        **gg_kwargs
    )
    if args.make_fws:
        from fireworks import Firework, LaunchPad

        launchpad = LaunchPad.auto_load()
        for fw in res:
            fw = Firework([res[fw]], name=fw)
            print(fw.name)
            launchpad.add_wf(fw)


if __name__ == "__main__":
    main()
