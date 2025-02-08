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

import os
import sys

import numpy as np
import yaml
from ase.parallel import paropen
from ase.units import Ha
from gpaw import restart
from pyscf.lib import chkfile
from ciderpress.gpaw.descriptors import get_descriptors
#import traceback

def get_exx(data_dir, calc, kpts, save_gap_data=False, run_exx=True):
    """
    :param save_dir:
    :param calc:
    :param p_be: (p_vbm, p_cbm), (s, k, n) for each
    :return:
    """
    from gpaw.hybrids.energy import non_self_consistent_energy

    if kpts is not None and run_exx: #mabdallah TODO: make sure kpts setting in get_exx makes sense
        calc.new(kpts=kpts)
    #calc.parameters.eigensolver
    # if solver is None:
    #    pass
    # elif not isinstance(solver, str) or solver.lower() == 'cg':
    #    calc.set(parallel={'domain': 1, 'band': 1})
    calc.get_potential_energy()
    if save_gap_data:
        from ase.dft.bandgap import bandgap

        gap, p_vbm, p_cbm = bandgap(calc)
        p_be = (p_vbm, p_cbm)
        print(p_be)
    else:
        p_be = None
    data = {}
    if run_exx:
        eterms = non_self_consistent_energy(calc, "EXX")
        data["exx"] = eterms[3:].sum() / Ha
    else:
        eterms = [Ha*calc.hamiltonian.e_total_free, Ha*-calc.hamiltonian.e_xc]
        data["exx"] = 0
    data["kpts"] = calc.parameters.kpts
    data["e_tot_orig"] = eterms[0] / Ha
    data["exc_orig"] = -eterms[1] / Ha
    data["xc_orig"] = calc.hamiltonian.xc.name
    if p_be is not None:
        if run_exx:
            from gpaw.hybrids.eigenvalues import non_self_consistent_eigenvalues as nsceigs
            eig_dft_dict = {k: {} for k in ["O", "U"]}
            vxc_dft_dict = {k: {} for k in ["O", "U"]}
            vxc_hyb_dict = {k: {} for k in ["O", "U"]}
            for l, p in zip(["O", "U"], p_be):
                eig_dft, vxc_dft, vxc_hyb = nsceigs(
                    calc, "EXX", n1=p[2], n2=p[2] + 1, kpt_indices=[p[1]]
                )
                eig_dft_dict[l][0] = eig_dft[p[0], 0, 0] / Ha
                vxc_dft_dict[l][0] = vxc_dft[p[0], 0, 0] / Ha
                vxc_hyb_dict[l][0] = vxc_hyb[p[0], 0, 0] / Ha
            data["eigvals"] = eig_dft_dict
            data["vxc_dft"] = vxc_dft_dict
            data["dval"] = vxc_hyb_dict
            data["p_be"] = p_be
        else:
            from gpaw.hybrids.eigenvalues import _semi_local
            eig_dft_dict = {k: {} for k in ["O", "U"]}
            vxc_dft_dict = {k: {} for k in ["O", "U"]}
            vxc_hyb_dict = {k: {} for k in ["O", "U"]}
            for l, p in zip(["O", "U"], p_be):
                eig_dft, vxc_dft, vxc_hyb = _semi_local(
                    calc, calc.hamiltonian.xc.name, n1=p[2], n2=p[2] + 1, kpt_indices=[p[1]]
                )
                eig_dft_dict[l][0] = eig_dft[p[0], 0, 0]
                vxc_dft_dict[l][0] = vxc_dft[p[0], 0, 0]
                vxc_hyb_dict[l][0] = vxc_hyb[p[0], 0, 0]
            data["eigvals"] = eig_dft_dict
            data["vxc_dft"] = vxc_dft_dict
            data["dval"] = vxc_hyb_dict
            data["p_be"] = p_be 
    if run_exx:
        with paropen(os.path.join(data_dir, "exx_data.yaml"), "w") as f:
            yaml.dump(data, f, Dumper=yaml.CDumper)
    else:
        return data
        #with paropen(os.path.join(data_dir, "data_no_exx.yaml"), "w") as f:
        #    yaml.dump(data, f, Dumper=yaml.CDumper)


def arr_to_strk(arr, nspin, p_be):
    if nspin == 2:
        v, c = (p_be[0][0], arr[0]), (p_be[1][0], arr[1])
    else:
        v, c = arr[0], arr[1]
    return {
        "O": {"0": v},
        "U": {"0": c},
    }


def intk_to_strk(d):
    if not isinstance(d, dict):
        return d
    nd = {}
    for k, v in d.items():
        nd[str(k)] = intk_to_strk(v)
    return nd

def find_working_qmax(calc, feat_settings, p_be, initial_qmax=300, step=100, max_qmax=2000):
    """
    Tries to find a working qmax value by incrementing from an initial value.

    :param calc: The calculator object.
    :param feat_settings: Feature settings for the descriptors.
    :param p_be: The p_be parameter for get_descriptors.
    :param initial_qmax: The starting qmax value.
    :param step: The increment step for qmax.
    :param max_qmax: The maximum qmax value to test.
    :return: A tuple of (working qmax, res, rho_res).
    """
    qmax = initial_qmax

    while qmax <= max_qmax:
        try:
            res = get_descriptors(calc, feat_settings, p_i=p_be, qmax=qmax)
            rho_res = get_descriptors(calc, "l", p_i=p_be, qmax=qmax)
            print(f"Successfully found working qmax value: {qmax}")
            return qmax, res, rho_res
        except Exception as e:
            print(f"Exception occurred for qmax {qmax}: {e}, trying next qmax")
            qmax += step

    raise RuntimeError(f"Failed to find a working qmax value up to {max_qmax}")

def save_features(save_file, data_dir, calc, feat_settings, save_gap_data=False, kpts_for_exx=None):
    exx_data_path = os.path.join(data_dir, "exx_data.yaml")
    if not os.path.exists(exx_data_path):
        print(f"EXX data file not found at {exx_data_path}, will extract data from calc.gpw (no EXX!)")
        data = get_exx(data_dir, calc, kpts_for_exx, save_gap_data=save_gap_data, run_exx=False)
        #with paropen(os.path.join(data_dir, "data_no_exx.yaml"), "r") as f:
        #    data = yaml.load(f, Loader=yaml.CLoader)
    else:
        print(f"Extracting data (including EXX) from {exx_data_path}")
        with paropen(exx_data_path, "r") as f:
            data = yaml.load(f, Loader=yaml.CLoader)

    
    #if data.get("exc_orig") is not None:
    #    raise Warning("exc_orig data found in exx_data.yaml, however, will extract from calc.gpw (should be same)")
    ##exc_orig = calc.hamiltonian.e_xc #unit: Ha already
    ##for setup in calc.hamiltonian.setups:
    ##    xcc = setup.xc_correction
    ##    if xcc is not None:
     ##       exc_orig += xcc.e_xc0
    ##data["exc_orig"] = exc_orig #mabdallah TODO: this is absolute energy NOT relative to spherical spin-non polarized atoms (GPAW default)
    data.pop("kpts")
    if save_gap_data:
        data["eigvals"] = intk_to_strk(data["eigvals"])
        data["vxc_dft"] = intk_to_strk(data["vxc_dft"])
        data["dval"] = intk_to_strk(data["dval"])
        p_be = data.pop("p_be")
    else:
        if "p_be" in data.keys():
            data.pop("p_be")
        p_be = None
    try:
        res = get_descriptors(calc, feat_settings, p_i=p_be)
        rho_res = get_descriptors(calc, "l", p_i=p_be)
    except Exception as e:
        print(f"Exception occurred in get_descriptors: {e}. Running qmax search...")  # Add this line to print the exception
        qmax, res, rho_res = find_working_qmax(calc, feat_settings, p_be)
   
    #last_qmax = None
    #last_error_type = None
    #qmax = 300
    #step = 500
    
    #while qmax <= 10000:
    #    try:
    #        res = get_descriptors(calc, feat_settings, p_i=p_be, qmax=qmax)
    #        rho_res = get_descriptors(calc, "l", p_i=p_be, qmax=qmax)
    #        print(f"Successfully found working qmax value: {qmax}")
    #        break
    #    except (RuntimeError, AssertionError) as e:
    #        print(f"\nException details:\n{traceback.format_exc()}\n")  # Print full traceback
    #        if isinstance(e, RuntimeError) and "NLDF exponent is too large" in str(e):
    #            error_type = "too_low"
    #            print(f"qmax {qmax} is too low, trying higher value...")
    #        elif isinstance(e, AssertionError):
    #            error_type = "too_high"
    #            print(f"qmax {qmax} is too high, adjusting...")
    #        else:
    #            print(f"Unexpected error for qmax {qmax}: {str(e)}")
    #        raise
            
    #    if last_error_type == "too_low" and error_type == "too_high":
    #        # We've jumped from too low to too high, try values in between
    #        qmax = last_qmax + (qmax - last_qmax) // 2
    #        step = (qmax - last_qmax) // 2
    #    else:
    #            last_qmax = qmax
    #            last_error_type = error_type
    #            qmax += step
            
    #    if step < 1:  # If we're making tiny adjustments and still failing
    #            raise RuntimeError(f"Could not find a working qmax value between {last_qmax} and {qmax}")
    #        continue
    #else:
    #    raise RuntimeError("Failed to find working qmax value between 300 and 10000")
    
    if p_be is None:
        feat_sig, all_wt = res
        rho_sig, _ = rho_res
    else:
        feat_sig, dfeat_jig, all_wt = res
        rho_sig, drho_jig, _ = rho_res
        data.update(
            {
                "ddesc": arr_to_strk(dfeat_jig, feat_sig.shape[0], p_be),
                "drho_data": arr_to_strk(drho_jig, feat_sig.shape[0], p_be),
            }
        )
    nspin = feat_sig.shape[0]
    data.update(
        {
            "rho_data": rho_sig,
            "desc": feat_sig,
            "wt": all_wt,
            "nspin": nspin,
        }
    )

    data["val"] = data["exx"] * np.ones_like(all_wt) / (nspin * all_wt.sum())

    if nspin == 2:
            data["val"] = np.stack([data["val"], data["val"]])  # sums to exx
    else:
        data["val"] = data["val"][np.newaxis, :]
    if calc.world.rank == 0:
        save_dir = os.path.dirname(os.path.abspath(save_file))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        chkfile.dump(save_file, "train_data", data)


def call_gpaw(settings_file=None, settings_dict=None):
    """Call GPAW with either file path or settings dictionary
    
    Args:
        settings_file (str, optional): Path to settings YAML file. If None, uses sys.argv[1]
        settings_dict (dict, optional): Direct settings dictionary. Takes precedence over file
    """
    if settings_dict is not None:
        settings = settings_dict
    else:
        if settings_file is None:
            settings_file = sys.argv[1]
        with paropen(settings_file, "r") as f:
            settings = yaml.load(f, Loader=yaml.Loader)

    data_dir = settings["data_dir"]
    task = settings["task"]  # should be EXX or FEAT
    feat_settings = settings["feat_settings"] #features settings object
    atoms, calc = restart(os.path.join(data_dir, "calc.gpw"), txt="-")
    if task == "EXX":
        get_exx(
            data_dir,
            calc,
            settings["kpts"],
            save_gap_data=settings.get("save_gap_data"),
        )
    elif task == "FEAT":
        save_features(
            settings["save_file"],
            data_dir,
            calc,
            feat_settings,
            save_gap_data=settings.get("save_gap_data"),
            kpts_for_exx=settings.get("kpts"),
        )


if __name__ == "__main__":
    call_gpaw()
