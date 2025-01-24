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
from argparse import ArgumentParser

import yaml
#from ciderpress.density import GG_AMIN #mabdallah: commented this
from orchard.gpaw_data_caller import call_gpaw
from ciderpress.dft.settings import (
    FracLaplSettings,
    HybridSettings,
    NLDFSettings,
    SDMXBaseSettings,
    SemilocalSettings,
)
from ciderpress.gpaw.descriptors import get_descriptors
from orchard.gpaw_tasks import StoreFeatures
#from orchard.workflow_utils import SAVE_ROOT, load_mol_ids
from orchard.workflow_utils import SAVE_ROOT, get_save_dir, load_mol_ids

def get_feat_type(settings):
    if settings == "l":
        return "REF"
    elif isinstance(settings, SemilocalSettings):
        return "SL"
    elif isinstance(settings, NLDFSettings):
        return "NLDF"
    elif isinstance(settings, FracLaplSettings):
        return "NLOF"
    elif isinstance(settings, SDMXBaseSettings):
        return "SDMX"
    elif isinstance(settings, HybridSettings):
        return "HYB"
    else:
        raise ValueError

def compile_dataset(
    feat_settings,
    feat_name,
    dataset_name,
    mol_id_list,
    save_root,
    functional,
    basis,
    save_gap_data=False,
    save_baselines=True,
    make_fws=False,
    skip_existing=False,
    save_dir=None,
):
    if not (isinstance (feat_settings, SemilocalSettings) or isinstance (feat_settings, NLDFSettings)):
        raise NotImplementedError("Only SL Settings and NLDF Settings are supported for GPAW currently.")
    if basis!="GPAW":
        raise ValueError("Only GPAW basis is supported for compile_gpaw_dataset. Check compile_pyscf_dataset for PySCF.")
   
  #  if save_gap_data:
  #      orbs = {"O": [0], "U": [0]}
  #  else:
  #      orbs = None
  #  orbs = None
    feat_type = get_feat_type(feat_settings)

    if save_dir is None:
        save_dir = os.path.join(
            save_root, "DATASETS", functional, basis, feat_type, feat_name
        )
    else:
        save_dir = os.path.join(save_dir, feat_type, feat_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    settings = {
        "DATASET_NAME": dataset_name,
        "FEAT_NAME": feat_name,
        "MOL_IDS": mol_id_list,
        "SAVE_ROOT": save_root,
        "FUNCTIONAL": functional,
        "BASIS": basis,
        "FEAT_SETTINGS": feat_settings,
        "SAVE_GAP_DATA": save_gap_data,
    }
    print(save_dir, save_root, feat_name)
    settings_fname = "{}_settings.yaml".format(dataset_name)
    print(os.path.join(save_dir, settings_fname))
    with open(os.path.join(save_dir, settings_fname), "w") as f:
        yaml.dump(settings, f)

    fwlist = {}

    for mol_id in mol_id_list:
        logging.info("Computing descriptors for {}".format(mol_id))
        data_dir = get_save_dir(save_root, "KS", basis, mol_id, functional) #mabdallah TODO: should check how to customize this, this should be location of gpw file
        save_file = os.path.join(save_dir, mol_id + ".hdf5")
        if os.path.exists(save_file) and skip_existing:
            print("Already exists, skipping:", mol_id)
            continue
        calc_settings = {
            "task": "FEAT",
            "data_dir": data_dir,
            "save_file": save_file,
            "save_gap_data": save_gap_data,
            "save_baselines": save_baselines,
            "feat_settings": feat_settings,
        } 
        if make_fws:
            fwname = "gpaw_feature_{}_{}".format(feat_name, mol_id)
            calc_settings["feat_settings"] = yaml.dump(calc_settings["feat_settings"], Dumper=yaml.CDumper) #check this 
            fwlist[fwname] = StoreFeatures(settings=calc_settings)
        else:
            call_gpaw(settings_dict=calc_settings) ##TODO: check if this is correct
    return fwlist


def compile_exx_dataset(
    MOL_IDS,
    SAVE_ROOT,
    FUNCTIONAL,
    kpt_density,
    save_gap_data=False,
    save_baselines=True,
):
    fwlist = {}

    for MOL_ID in MOL_IDS:
        logging.info("Computing exx for {}".format(MOL_ID))
        data_dir = os.path.join(SAVE_ROOT, "KS", FUNCTIONAL, MOL_ID)
        new_kpts = (
            None
            if "magmom" in MOL_ID
            else {"density": kpt_density, "even": True, "gamma": True}
        )
        nproc = 1 if "magmom" in MOL_ID else None
        calc_settings = {
            "task": "EXX",
            "kpts": new_kpts,
            "nproc": nproc,
            "encut": 520,
            "data_dir": data_dir,
            "save_gap_data": save_gap_data,
            "save_baselines": save_baselines,
        }
        fwname = "gpaw_exx_{}".format(MOL_ID)
        fwlist[fwname] = StoreFeatures(settings=calc_settings)

    return fwlist


def main():
    logging.basicConfig(level=logging.INFO)

    m_desc = "Setup FWs to compile dataset of XC descriptors with GPAW"

    parser = ArgumentParser(description=m_desc)
    parser.add_argument(
        "mol_id_file", type=str, help="yaml file from which to read mol_ids to parse"
    )
    parser.add_argument(
        "feat_name",
        type=str,
        help="Name of the feature set being generated, used to make "
        "save directory for generated data.",
    )
    parser.add_argument(
        "basis",
        metavar="basis",
        type=str,
        help="Basis set that was used for the DFT calculations",
    )
    parser.add_argument(
        "--settings-file",
        metavar="settings_file",
        type=str,
        default=None,
        help="Path to a yaml file containing a serialized FeatureSettings "
        "class. If not provided, generates the reference data "
        "(i.e. semilocal density, EXX and XC reference, etc.)",
    )
    parser.add_argument(
        "--functional",
        metavar="functional",
        type=str,
        default=None,
        help="exchange-correlation functional, HF for Hartree-Fock",
    )

    parser.add_argument(
        "--make-fws",
        action="store_true",
        help="If True, make a firework to generate features for each"
        "molecule, to be run later. If False, generate features"
        "for each molecule serially within this script.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip system if save_file exists already",
    )

    parser.add_argument(
        "--save-gap-data",
        action="store_true",
        help="If True, store the band gap data for each molecule.",
    )

    parser.add_argument("--exx-only", action="store_true")
    parser.add_argument("--kpt-density", default=4.5, type=float)
    parser.add_argument(
        "--save-dir",
        default=None,
        type=str,
        help="override default save directory for features",
    )
    args = parser.parse_args()
    if args.settings_file is None or args.settings_file == "__REF__":
        settings = "l"
    else:
        with open(args.settings_file, "r") as f:
            settings = yaml.load(f, Loader=yaml.CLoader)

    mol_ids = load_mol_ids(args.mol_id_file)
    if args.mol_id_file.endswith(".yaml"):
        mol_id_code = args.mol_id_file[:-5]
    else:
        mol_id_code = args.mol_id_file
    if args.exx_only:
        res = compile_exx_dataset(
            mol_ids,
            SAVE_ROOT,
            args.functional,
            kpt_density=args.kpt_density,
            save_gap_data=args.save_gap_data,
        )
    else:
        res = compile_dataset(
            settings,
            args.feat_name,
            mol_id_code.upper().split("/")[-1],
            mol_ids,
            SAVE_ROOT,
            args.functional,
            args.basis,
            save_gap_data=args.save_gap_data,
            make_fws=args.make_fws,
            skip_existing=args.skip_existing,
            save_dir=args.save_dir,
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
