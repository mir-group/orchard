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

import importlib
import os
import sys
import time
import traceback
import warnings
from argparse import ArgumentParser

import numpy as np
import pickle
import yaml
from ciderpress.dft.settings import LDA_FACTOR, FeatureSettings
from ciderpress.dft.transform_data import FeatureList
from ciderpress.dft.baselines import BASELINE_CODES
from ciderpress.models.dft_kernel import DFTKernel, DFTKernel2 
from ciderpress.models.train import MOLGP, MOLGP2, strk_to_tuplek
from joblib import dump, load
from orchard.workflow_utils import load_rxns
from orchard.sparsify import select_kmeans_pool, count_total_samples
from scipy.linalg import cho_solve, cholesky

"""
Dataset file format:
systems:
  <dataset_name>:
    inverse_sampling_density: <integer>
    load_orbs: <bool>
  ...
reactions:
  <rxn_dataset_name0>: <integer 0 (X), 1 (C), or 2 (XC)>
  <rxn_dataset_name1>: <integer 0 (X), 1 (C), or 2 (XC)>
  ...
"""


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def _get_name_dict(args):
    return {
        "REF": args.ref_feat_name,
        "SL": args.sl_feat_name,
        "NLDF": args.nldf_feat_name,
        "NLOF": args.nlof_feat_name,
        "SDMX": args.sdmx_feat_name,
        "HYB": args.hyb_feat_name,
    }


def _get_val_name_dict(args):
    """Get validation feature names, defaulting to training names if not specified."""
    return {
        "REF": args.val_ref_feat_name or args.ref_feat_name,
        "SL": args.val_sl_feat_name or args.sl_feat_name,
        "NLDF": args.val_nldf_feat_name or args.nldf_feat_name,
        "NLOF": args.val_nlof_feat_name or args.nlof_feat_name,
        "SDMX": args.val_sdmx_feat_name or args.sdmx_feat_name,
        "HYB": args.val_hyb_feat_name or args.hyb_feat_name,
    }


def write_train_analysis(gpr, rxn_id_list, fname="train_analysis.yaml"):
    K = gpr.Kcov_
    y = gpr.y_mol_
    alpha = gpr.alpha_mol_
    y_pred = K.dot(alpha)
    y_pred - y
    rtkd = np.sqrt(np.diag(K))
    np.array(rxn_id_list)
    nitems = 20
    for i, rxn_id in enumerate(rxn_id_list):
        rel_cov = K[i] / (rtkd * rtkd[i] + 1e-16)
        # print(rxn_id)
        inds = np.argsort(np.abs(rel_cov))
        inds = np.flip(inds)[:nitems]
        # print(rxn_id_arr[inds])
        # print(rel_cov[inds])
        # print(dy[inds])
        # print()
    ana_set = {
        "K": K,
        "Kfull": gpr.K_,
        "y_pred": y_pred,
        "y": y,
        "alpha": alpha,
        "rxn_id_list": rxn_id_list,
    }
    with open(fname, "w") as f:
        yaml.dump(ana_set, f, Dumper=yaml.CDumper)


def write_validation_analysis(gpr, val_rxn_list, val_rxn_id_list, 
                            val_molid_map, args, data_settings, 
                            mapping_plans=None, val_args=None, fname="validation_analysis.yaml"):
    """Write validation analysis for a trained model.
    
    Args:
        gpr: Trained MOLGP model
        val_rxn_list: List of validation reactions
        val_rxn_id_list: List of validation reaction IDs
        val_molid_map: Dictionary mapping dataset names to molecule IDs
        args: Command line arguments
        data_settings: Data settings dictionary
        mapping_plans: Optional mapping plans for raw output computation
        val_args: Optional validation-specific args (defaults to args)
        fname: Output filename
    """
    # Use validation args if provided, otherwise use regular args
    if val_args is None:
        val_args = args
    # Get number of training reactions
    n_train = len(gpr.rxn_ref_list) - len(val_rxn_list)
    
    # Combine kernel contributions to compute validation-training covariance
    val_K = np.zeros((len(val_rxn_list), n_train))
    for kernel in gpr.kernels:
        # Get covariances
        Kmn_val = np.stack(kernel.rxn_cov_list[-len(val_rxn_list):])  # validation x control
        Kmn_train = np.stack(kernel.rxn_cov_list[:n_train])  # training x control
        Kmm = kernel.get_kctrl()  # control x control
        
        # Compute K_val,train = K_val,ctrl @ K_ctrl,ctrl^{-1} @ K_ctrl,train
        M = Kmm.shape[0]
        mini_noise = gpr.numerical_epsilon * np.identity(M)
        L = cholesky(Kmm + mini_noise, lower=True)
        
        # Solve for K_ctrl,ctrl^{-1} @ K_ctrl,train
        Kimn_train = cho_solve((L, True), Kmn_train.T)  # control x training
        
        # Compute validation-training covariance contribution
        val_K += Kmn_val.dot(Kimn_train)
    
    # Compute predictions
    y_val_pred = val_K.dot(gpr.alpha_mol_)
    
    # Get true values from the stored reference list
    # These were computed by gpr.add_reactions()
    y_val = np.array(gpr.rxn_ref_list[-len(val_rxn_list):])
    
    # Compute raw GP outputs for validation molecules if mapping available
    mol_raw_outputs = {}
    if mapping_plans and all(mfunc is not None for mfunc in mapping_plans):
        mapped_xc = gpr.map(mapping_plans)
        
        for dset_name in val_molid_map:
            mol_ids = val_molid_map[dset_name]
            
            for mol_id in mol_ids:
                try:
                    # Use validation-specific feature directories
                    val_ddirs = find_datasets(dset_name, val_args, data_settings)
                    data = MOLGP.load_data(val_ddirs, mol_id, get_orb_deriv=False)
                    desc = data["desc"]
                    weights = data["wt"]
                    X0T_norm = desc
                    
                    _, _, f_raw, _ = mapped_xc(X0T_norm, rhocut=0, 
                                             return_raw_ml_output=True)
                    
                    # Handle individual kernel outputs
                    if len(gpr.kernels) > 1:
                        # f_raw has shape (num_kernels, ...), process each kernel separately
                        kernel_outputs = []
                        for k_idx in range(len(gpr.kernels)):
                            f_k = f_raw[k_idx]  # Extract this kernel's output
                            
                            if f_k.ndim == 2:  # SEP mode
                                f_avg_k = (f_k * weights).sum(axis=1) / weights.sum()
                                kernel_outputs.append({
                                    "kernel_index": k_idx,
                                    "f_avg_per_spin": f_avg_k.tolist(),
                                    "f_avg_total": f_avg_k.sum(),
                                    "mode": "SEP"
                                })
                            else:  # NPOL/POL mode
                                f_avg_k = (f_k * weights).sum() / weights.sum()
                                kernel_outputs.append({
                                    "kernel_index": k_idx,
                                    "f_avg": float(f_avg_k),
                                    "mode": "NPOL/POL"
                                })
                        
                        # Also compute total (summed) output
                        f_raw_total = f_raw.sum(axis=0)
                        if f_raw_total.ndim == 2:
                            f_avg_total = (f_raw_total * weights).sum(axis=1) / weights.sum()
                            total_output = {
                                "f_avg_per_spin": f_avg_total.tolist(),
                                "f_avg_total": f_avg_total.sum(),
                                "mode": "SEP"
                            }
                        else:
                            f_avg_total = (f_raw_total * weights).sum() / weights.sum()
                            total_output = {
                                "f_avg": float(f_avg_total),
                                "mode": "NPOL/POL"
                            }
                        
                        mol_raw_outputs[mol_id] = {
                            "individual_kernels": kernel_outputs,
                            "total_output": total_output,
                            "num_kernels": len(gpr.kernels)
                        }
                    else:
                        # Single kernel case
                        if f_raw.ndim == 2:
                            f_avg = (f_raw * weights).sum(axis=1) / weights.sum()
                            mol_raw_outputs[mol_id] = {
                                "f_avg_per_spin": f_avg.tolist(),
                                "f_avg_total": f_avg.sum(),
                                "mode": "SEP",
                                "num_kernels": 1
                            }
                        else:
                            f_avg = (f_raw * weights).sum() / weights.sum()
                            mol_raw_outputs[mol_id] = {
                                "f_avg": float(f_avg),
                                "mode": "NPOL/POL",
                                "num_kernels": 1
                            }
                except Exception as e:
                    print(f"Warning: Could not compute raw output for {mol_id}: {e}")
                    mol_raw_outputs[mol_id] = {"error": str(e)}
    
    # Get validation feature names used
    val_name_dict = _get_val_name_dict(val_args)
    feature_names = {
        feat_type: feat_name 
        for feat_type, feat_name in val_name_dict.items() 
        if feat_name is not None
    }
    
    # Save validation analysis
    ana_set = {
        "K_val_train": val_K,
        "y_pred": y_val_pred,
        "y": y_val,
        "rxn_id_list": val_rxn_id_list,
        "mol_raw_outputs": mol_raw_outputs,
        "feature_names": feature_names,
        "validation_datasets": list(val_molid_map.keys()),
    }
    with open(fname, "w") as f:
        yaml.dump(ana_set, f, Dumper=yaml.CDumper)


def get_base_path(dset_name, data_settings):
    pathid = data_settings["systems"][dset_name]["path"]
    if isinstance(pathid, int):
        base_dname = data_settings["paths"][pathid]
    else:
        base_dname = pathid
    return base_dname


def parse_settings(set0, data_settings, args):
    base_dname = get_base_path(set0, data_settings)
    settings_dict = {}
    name_dict = _get_name_dict(args)
    for feat_type, feat_name in name_dict.items():
        if feat_name is None:
            settings_dict[feat_type] = None
            continue
        dname = os.path.join(
            base_dname,
            feat_type,
            feat_name,
        )
        fname = os.path.join(dname, "{}_settings.yaml".format(set0))
        print(f"Loading settings from: {fname}")
        with open(fname, "r") as f:
            settings_dict[feat_type] = yaml.load(f, Loader=yaml.CLoader)[
                "FEAT_SETTINGS"
            ]
    if args.normalizer_file is None:
        normalizers = None
    else:
        with open(args.normalizer_file, "r") as f:
            normalizers = yaml.load(f, Loader=yaml.CLoader)
    settings = FeatureSettings(
        sl_settings=settings_dict["SL"],
        nldf_settings=settings_dict["NLDF"],
        nlof_settings=settings_dict["NLOF"],
        sdmx_settings=settings_dict["SDMX"],
        hyb_settings=settings_dict["HYB"],
        normalizers=normalizers,
    )
    print(f"SL: {settings_dict['SL']}")
    if args.normalizer_file is None:
        settings.assign_reasonable_normalizer()
        with open("__norms.yaml", "w") as f:
            yaml.dump(settings.normalizers, f)
    return settings


def find_datasets(dataset_name, args, data_settings):
    name_dict = _get_name_dict(args)
    ddirs = {}
    for feat_type, feat_name in name_dict.items():
        if feat_name is None:
            ddirs[feat_type] = None
            continue
        base_path = get_base_path(dataset_name, data_settings)
        fname = "{}_settings.yaml".format(dataset_name)
        fname = os.path.join(base_path, feat_type, feat_name, fname)
        if not os.path.exists(fname):
            raise FileNotFoundError("Data directory {} does not exist.".format(fname))
        ddirs[feat_type] = os.path.dirname(fname)
    return ddirs


def get_plan_module(plan_file):
    if plan_file.startswith("@"):
        plan_module = importlib.import_module(plan_file[1:])
    else:
        if not os.path.exists(plan_file):
            print(f"ERROR: Plan file not found at: {plan_file}")
            raise FileNotFoundError(f"Plan file not found at: {plan_file}")
        spec = importlib.util.spec_from_file_location("plan_module", plan_file)
        plan_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plan_module)
    return plan_module


def parse_dataset_for_ctrl(fname, n, args, data_settings, feat_settings):
    print(fname, n, data_settings)
    dirnames = find_datasets(fname, args, data_settings)
    with open(os.path.join(dirnames["SL"], "{}_settings.yaml".format(fname)), "r") as f:
        settings = yaml.load(f, Loader=yaml.CLoader)
        mol_ids = settings["MOL_IDS"]
    Xlist = []
    GXOlist = []
    GXUlist = []
    ylist = []
    
    # Debug: Print available directories
    print("\nAvailable directories:")
    for key, path in dirnames.items():
        print(f"{key}: {path}")
    
    total_after_cutoff = 0
    total_after_stride = 0
    for mol_id in mol_ids:
        print(f"\nProcessing molecule: {mol_id}")
        
        # Debug: Load and inspect each feature file before MOLGP
        for feat_type in ["REF", "SL", "NLDF", "NLOF", "SDMX", "HYB"]:
            if dirnames.get(feat_type):
                feat_file = os.path.join(dirnames[feat_type], mol_id + ".hdf5")
                if os.path.exists(feat_file):
                    from pyscf.lib import chkfile
                    data = chkfile.load(feat_file, "train_data")
                    print(f"\n{feat_type} file contents:")
                    print(f"Keys: {list(data.keys())}")
                    for key, value in data.items():
                        if isinstance(value, np.ndarray):
                            print(f"{key} shape: {value.shape}")
                        else:
                            print(f"{key} type: {type(value)}")
        
        # Now call the original MOLGP load_data
        data = MOLGP.load_data(dirnames, mol_id, None)
        from pyscf.lib import chkfile #mabdallah TODO: Quick fix for mismatch. Should investigate CIDER at somne point...
        sl_data = chkfile.load(os.path.join(dirnames["SL"], mol_id + ".hdf5"), "train_data")    
        # Per-spin density mask (nspin, nsamp)
        cond_spin = data["desc"][:, 0, :] > args.density_cutoff
        spin_counts = cond_spin.sum(axis=1)
        print(
            f"[{fname}::{mol_id}] per-spin surviving points "
            f"{spin_counts.tolist()} / {cond_spin.shape[1]} (cutoff {args.density_cutoff})"
        )
        print(data["desc"].shape, data["val"].shape)
        # Compute y using per-spin mask (2D boolean)
        y = (
            data["val"][cond_spin]
            / (LDA_FACTOR * data["desc"][:, 0][cond_spin] ** (4.0 / 3))
            - 1
        )
        #y = sl_data["val"][cond] / (LDA_FACTOR * data["desc"][:, 0][cond] ** (4.0 / 3)) - 1 this is fix for mismatch
        # Select columns (grid points) if ANY spin channel passes the density threshold
        cond = np.any(cond_spin, axis=0)
        print(
            f"[{fname}::{mol_id}] surviving grid points after any-spin: "
            f"{int(cond.sum())}/{cond.size}"
        )
        desc = data["desc"][:, :, cond]
        X = feat_settings.normalizers.get_normalized_feature_vector(desc)
        nsamp_after_cutoff = X.shape[-1]
        total_after_cutoff += nsamp_after_cutoff
        if "ddesc" in data:
            ddesc = strk_to_tuplek(data["ddesc"])
            print(ddesc.keys())
            has_ddesc = True
            GXO = ddesc[("O", 0)][1][:, cond]
            GXU = ddesc[("U", 0)][1][:, cond]
            GXO = feat_settings.normalizers.get_derivative_of_normed_features(
                desc[ddesc[("O", 0)][0]], GXO
            )
            GXU = feat_settings.normalizers.get_derivative_of_normed_features(
                desc[ddesc[("U", 0)][0]], GXU
            )
        else:
            has_ddesc = False
        if args.randomize:
            inds = np.arange(X.shape[-1])
            np.random.shuffle(inds)
            X = X[..., inds]
            if has_ddesc:
                GXO = GXO[..., inds]
                GXU = GXU[..., inds]
        if n is not None:
            X_down = X[..., ::n]
            nsamp_after_stride = X_down.shape[-1]
            total_after_stride += nsamp_after_stride
            Xlist.append(X_down)
            if has_ddesc:
                GXOlist.append((ddesc[("O", 0)][0], GXO[..., ::n]))
                GXUlist.append((ddesc[("U", 0)][0], GXU[..., ::n]))
            ylist.append(y[::n])
        else:
            total_after_stride += nsamp_after_cutoff
            Xlist.append(X)
            if has_ddesc:
                GXOlist.append((ddesc[("O", 0)][0], GXO))
                GXUlist.append((ddesc[("U", 0)][0], GXU))
            ylist.append(y)
    total_samples = count_total_samples(Xlist)
    print(f"[{fname}] total samples after density cutoff: {total_after_cutoff:,}")
    if n is not None:
        print(f"[{fname}] total samples after random stride (n={n}): {total_after_stride:,}")
    else:
        print(f"[{fname}] stride disabled; samples passed to pooling: {total_after_stride:,}")
    print(f"[{fname}] total samples after parsing (pre-sparsify): {total_samples:,}")
    return Xlist, GXOlist, GXUlist, ylist, mol_ids


def get_fd_x1(kernel, Xlist, DXlist, delta=1e-5):
    if len(Xlist) == 0:
        return 0
    nfeat = Xlist[0].shape[1]
    print("NFEAT", nfeat)
    deriv = 0
    for i in range(nfeat):
        slist = [DX[0] for DX in DXlist]
        IDXlist = [DX[1][i] for DX in DXlist]
        for s, X in zip(slist, Xlist):
            X[s, i, :] += 0.5 * delta
        utmp = kernel.X0Tlist_to_X1array_mul(Xlist, IDXlist)
        for s, X in zip(slist, Xlist):
            X[s, i, :] -= delta
        ltmp = kernel.X0Tlist_to_X1array_mul(Xlist, IDXlist)
        for s, X in zip(slist, Xlist):
            X[s, i, :] += 0.5 * delta
        print(i, utmp[0], ltmp[0])
        deriv += (utmp - ltmp) / delta
    return deriv


def analyze_cov(X1, avg_and_std=None):
    if avg_and_std is None:
        avg = np.mean(X1, axis=0)
        std = np.std(X1, axis=0)
    else:
        avg, std = avg_and_std
    XW = X1 - avg
    XW /= std
    cov = XW.T.dot(XW) / XW.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    # return avg, std, evals, evecs
    print("COV")
    print(avg)
    print(std)
    print(cov)
    print(evals)
    print(evecs)
    return avg, std, cov, evals, evecs


def main():
    parser = ArgumentParser(
        description="Fits a GP exchange(-correlation) model to "
        "molecular/solid-state energy differences and "
        "orbital energies."
    )

    parser.add_argument("save_file", type=str, help="file to which to save new GP")
    parser.add_argument("ref_feat_name", type=str, help="Name of reference data set")
    parser.add_argument("sl_feat_name", type=str, help="Name of semilocal feature set")
    parser.add_argument(
        "--normalizer-file",
        type=str,
        default=None,
        help="Path to normalizer yaml file.",
    )
    parser.add_argument(
        "--nldf-feat-name", type=str, default=None, help="Name of NLDF feature set."
    )
    parser.add_argument(
        "--nlof-feat-name", type=str, default=None, help="Name of NLOF feature set."
    )
    parser.add_argument(
        "--sdmx-feat-name", type=str, default=None, help="Name of SDMX feature set."
    )
    parser.add_argument(
        "--hyb-feat-name",
        type=str,
        default=None,
        help="Name of hybrid DFT feature set.",
    )
    parser.add_argument(
        "--kernel-plan-file",
        type=str,
        help="Settings file for list of kernels. See "
        "ciderpress.models.kernel_plans.settings_example.yaml "
        "for documentation and format.",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        help="Path to yaml file containing names of datasets to load "
        "along with instructions for loading said datasets.",
    )
    parser.add_argument(
        "--kmeans-pool-size",
        type=int,
        default=20000,
        help="Default target size when dataset config enables k-means pooling.",
    )
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=8192,
        help="Batch size for scaler fitting and MiniBatchKMeans updates.",
    )
    parser.add_argument(
        "--kmeans-epochs",
        type=int,
        default=3,
        help="Number of streaming passes over each dataset when using k-means pooling.",
    )
    parser.add_argument(
        "--sparsify-debug-dir",
        type=str,
        default="/n/holystore01/LABS/kozinsky_lab/Lab/User/mabdallah/CIDER_gpaw/debug_sparsify",
        help="Optional directory where a debug snapshot of the unsparsified "
        "features is written for datasets that enable k-means pooling.",
    )
    parser.add_argument("-c", "--density-cutoff", type=float, default=1e-6)
    parser.add_argument("-s", "--seed", help="random seed", default=0, type=int)
    parser.add_argument(
        "-d",
        "--delete-k",
        action="store_true",
        help="Delete L (LL^T=K the kernel matrix) to save disk "
        "space. Need to refit when reloading to calculate "
        "covariance.",
    )
    parser.add_argument(
        "--nmax-sparse",
        type=int,
        default=None,
        help="If set, not more than this many points used in sparse set.",
    )
    parser.add_argument(
        "--control-tol",
        type=float,
        default=-1e-5,
        help="Reduce control point size for given tol. "
        "Negative value means to ignore.",
    )
    parser.add_argument(
        "--mol-sigma",
        type=float,
        default=0.03,
        help="Standard deviation noise parameter for total molecular energy data.",
    )
    parser.add_argument("--scale-override", type=float, default=None)
    parser.add_argument("--scale-mul", type=float, default=1.0)
    parser.add_argument(
        "--length-scale-mul",
        type=float,
        nargs="+",
        default=[1.0],
        help="Used for automatic length-scale initial guess.",
    )
    parser.add_argument(
        "--min-lscale",
        type=float,
        default=None,
        help="Minimum length-scale for GP kernel.",
    )
    parser.add_argument(
        "--libxc-baseline",
        type=str,
        default=None,
        help="Baseline libxc functional for the full model",
    )
    parser.add_argument(
        "--mapped-fname",
        type=str,
        default=None,
        help="If not None, map model and same to this file.",
    )
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--debug-model", type=str, default=None, help="Load joblib and print debug"
    )
    parser.add_argument(
        "--debug-spline", type=str, default=None, help="Load joblib and print debug"
    )
    parser.add_argument(
        "--reload-model",
        type=str,
        default=None,
        help="If path exists, load this model and refit (possibly with new "
        "weights on datasets) while ignoring other parameters.",
    )
    parser.add_argument(
        "--version2",
        action="store_true",
    )
    parser.add_argument(
        "--validation-dataset-file",
        type=str,
        default=None,
        help="Path to yaml file containing names of validation datasets "
        "on which to evaluate the trained model.",
    )
    parser.add_argument(
        "--val-ref-feat-name",
        type=str,
        default=None,
        help="Name of reference data set for validation (defaults to training ref name)",
    )
    parser.add_argument(
        "--val-sl-feat-name",
        type=str,
        default=None,
        help="Name of semilocal feature set for validation (defaults to training sl name)",
    )
    parser.add_argument(
        "--val-nldf-feat-name",
        type=str,
        default=None,
        help="Name of NLDF feature set for validation (defaults to training nldf name)",
    )
    parser.add_argument(
        "--val-nlof-feat-name",
        type=str,
        default=None,
        help="Name of NLOF feature set for validation (defaults to training nlof name)",
    )
    parser.add_argument(
        "--val-sdmx-feat-name",
        type=str,
        default=None,
        help="Name of SDMX feature set for validation (defaults to training sdmx name)",
    )
    parser.add_argument(
        "--val-hyb-feat-name",
        type=str,
        default=None,
        help="Name of hybrid DFT feature set for validation (defaults to training hyb name)",
    )
    args = parser.parse_args()
    if args.debug_model is not None:
        args.debug_model = load(args.debug_model)
    if args.debug_spline is not None:
        args.debug_spline = load(args.debug_spline)

    with open(args.dataset_file, "r") as f:
        data_settings = yaml.load(f, Loader=yaml.CLoader)
    with open(args.kernel_plan_file, "r") as f:
        kernel_plans = yaml.load(f, Loader=yaml.CLoader)

    np.random.seed(args.seed)
    datasets_list = list(data_settings["systems"].keys())
    settings = parse_settings(datasets_list[0], data_settings, args)
    print(
        "USPS", settings.get_feat_usps(), settings.get_feat_usps(with_normalizers=True)
    )
    print("UEGS", settings.ueg_vector(), settings.ueg_vector(with_normalizers=True))

    reload_bool = args.reload_model is not None and os.path.exists(args.reload_model)
    if reload_bool:
        gpr = load(args.reload_model)
        gpr.default_noise = args.mol_sigma
        ylist = []
        molid_map = {}
        for dset_name in datasets_list:
            dirnames = find_datasets(dset_name, args, data_settings)
            with open(
                os.path.join(dirnames["SL"], "{}_settings.yaml".format(dset_name)), "r"
            ) as f:
                settings = yaml.load(f, Loader=yaml.CLoader)
                mol_ids = settings["MOL_IDS"]
            molid_map[dset_name] = mol_ids
    else:
        Xlist = []
        GXRlist = []
        GXOlist = []
        GXUlist = []
        ylist = []
        molid_map = {}
        any_use_kmeans = False
        requested_pool_total = 0  # Sum pool targets from YAML across datasets that enable k-means
        for dset_name in datasets_list:
            sys_cfg = data_settings["systems"][dset_name]
            raw_kmeans = sys_cfg.get("kmeans_pool_size")
            if isinstance(raw_kmeans, bool):
                use_kmeans = raw_kmeans
                pool_target = args.kmeans_pool_size if use_kmeans else None
            elif raw_kmeans is None:
                use_kmeans = False
                pool_target = None
            else:
                use_kmeans = True
                pool_target = int(raw_kmeans)
            if use_kmeans and pool_target is None:
                pool_target = args.kmeans_pool_size
            # Keep stride sampling even when k-means pooling is enabled
            n = sys_cfg.get("inverse_sampling_density")
            (
                Xlist_tmp,
                GXOlist_tmp,
                GXUlist_tmp,
                y_tmp,
                dset_ids,
            ) = parse_dataset_for_ctrl(dset_name, n, args, data_settings, settings)
            if use_kmeans:
                before = count_total_samples(Xlist_tmp)
                print(
                    f"[{dset_name}] k-means pooling enabled; "
                    f"{before:,} samples before pooling.",
                    flush=True,
                )
                debug_dir = args.sparsify_debug_dir
                if debug_dir:
                    os.makedirs(debug_dir, exist_ok=True)
                    dbg_path = os.path.join(
                        debug_dir, f"{dset_name}_raw_features.pkl"
                    )
                    with open(dbg_path, "wb") as fh:
                        pickle.dump(
                            {
                                "data": [arr.copy() for arr in Xlist_tmp],
                                "mol_ids": dset_ids,
                            },
                            fh,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    print(
                        f"[{dset_name}] wrote debug snapshot to {dbg_path} (overwrite)",
                        flush=True,
                    )
                # Defer pooling until kernels are available
                any_use_kmeans = True
                # Accumulate requested pool size from YAML (fallback to CLI if YAML was boolean True)
                requested_pool_total += int(pool_target)
            Xlist += Xlist_tmp
            if len(GXOlist_tmp) == len(Xlist_tmp):
                GXRlist += Xlist_tmp
                GXOlist += GXOlist_tmp
                GXUlist += GXUlist_tmp
            ylist += y_tmp
            molid_map[dset_name] = dset_ids
        yctrl = np.concatenate(ylist, axis=0)

        kernels = []
        args.plan_files = []
        mapping_plans = []
        for plan in kernel_plans:
            plan_file = plan.pop("plan_file")
            plan_module = get_plan_module(plan_file)
            args.plan_files.append(plan_file)
            feature_list = FeatureList.load(plan["feature_list"])
            ctrl_tol = plan.get("ctrl_tol") or 1e-5
            ctrl_nmax = plan.get("ctrl_nmax")
            kcls = DFTKernel2 if args.version2 else DFTKernel
            if kcls == DFTKernel:
                mb = BASELINE_CODES[plan["multiplicative_baseline"]]
                ab = BASELINE_CODES.get(plan["additive_baseline"])
            else:
                mb = plan["multiplicative_baseline"]
                ab = plan.get("additive_baseline")
            kernels.append(
                kcls(
                    None,
                    feature_list,
                    plan["mode"],
                    mb,
                    additive_baseline=ab,
                    ctrl_tol=ctrl_tol,
                    ctrl_nmax=ctrl_nmax,
                    component=plan.get("component"),
                )
            )
            if "lscale_override" in plan: #mabdallah TODO: temporary fix
                lscale = np.array(plan.pop("lscale_override"))
                val_pca = None
                deriv_pca = None
            else:
                X1 = kernels[-1].X0Tlist_to_X1array(Xlist)
                #DXO1 = get_fd_x1(kernels[-1], GXRlist, GXOlist)
                #DXU1 = get_fd_x1(kernels[-1], GXRlist, GXUlist)
                #val_pca = analyze_cov(X1)
                #analyze_cov(DXO1, avg_and_std=val_pca[:2])
                #analyze_cov(DXU1, avg_and_std=val_pca[:2])
                #deriv_pca = analyze_cov(DXU1 - DXO1, avg_and_std=val_pca[:2])
                val_pca = None
                deriv_pca = None
                if X1.ndim == 2:
                    lscale = np.std(X1, axis=0)
                else:
                    lscale = np.std(X1, axis=(0, 1))
                # print("SHAPES", X1.shape, yctrl.shape)
            if "scale_override" in plan:
                scale = np.array(plan.pop("scale_override"))
            elif args.scale_override is None:
                scale = np.var(yctrl)
            else:
                scale = args.scale_override
            kernel = plan_module.get_kernel(
                natural_scale=scale,
                natural_lscale=lscale,
                scale_factor=args.scale_mul,
                lscale_factor=args.length_scale_mul,
                #val_pca=val_pca,
                #deriv_pca=deriv_pca,
            )
            kernels[-1].set_kernel(kernel)
            if "mapping_plan" in dir(plan_module):
                mfunc = plan_module.mapping_plan
            else:
                mfunc = None
            mapping_plans.append(mfunc)
        gpcls = MOLGP2 if args.version2 else MOLGP
        gpr = gpcls(
            kernels,
            settings,
            libxc_baseline=args.libxc_baseline,
            default_noise=args.mol_sigma,
        )
        gpr.args = args
        # Apply global k-means pooling if any dataset requested it,
        # now that kernels (descriptor mapping) are available.
        if any_use_kmeans:
            before = count_total_samples(Xlist)
            print(
                f"[GLOBAL] applying k-means pooling before Nyström: {before:,} samples",
                flush=True,
            )
            start = time.time()
            # Choose pool size from YAML if provided; otherwise use CLI default
            pool_size = requested_pool_total if requested_pool_total > 0 else args.kmeans_pool_size
            print(
                f"[GLOBAL] desired pool size from YAML sum = {requested_pool_total or 0}; "
                f"using pool_size = {pool_size}",
                flush=True,
            )
            if before < pool_size:
                print(
                    f"[GLOBAL] warning: pool_size ({pool_size}) > candidates ({before}); "
                    f"reducing pool_size to {before}",
                    flush=True,
                )
                pool_size = before
            Xlist = select_kmeans_pool(
                Xlist,
                pool_size=pool_size,
                batch_size=args.kmeans_batch_size,
                kmeans_epochs=args.kmeans_epochs,
                random_state=args.seed,
                logger=lambda msg: print(f"[GLOBAL] {msg}", flush=True),
                progress_every=10000000,
                kernel=kernels[0],
            )
            after = count_total_samples(Xlist)
            print(
                f"[GLOBAL] k-means pooling complete: {before:,} -> {after:,} samples in {time.time() - start:.1f}s",
                flush=True,
            )

        gpr.set_control_points(Xlist, reduce=True)
        print("CTRL SIZE", [k.X1ctrl.shape for k in kernels])

    rxn_list = []
    rxn_id_list = []
    rxn_ids = list(data_settings["reactions"].keys())
    for i, rxn_id in enumerate(rxn_ids):
        rxn_dict = load_rxns(rxn_id)
        rxn_settings = data_settings["reactions"][rxn_id]
        mode = rxn_settings.get("mode") or 0
        for k, v in list(rxn_dict.items()):
            v.update(rxn_settings)
            rxn_id_list.append(k)
            rxn_list.append((mode, v))

    if reload_bool:
        gpr.reset_reactions()
    else:
        for i, fname in enumerate(datasets_list):
            load_orbs = data_settings["systems"][fname].get("load_orbs")
            mol_ids = molid_map[fname]
            fnames = find_datasets(fname, args, data_settings)
            gpr.store_mol_covs(
                fnames, mol_ids, get_orb_deriv=load_orbs, get_correlation=True
            )

    gpr.add_reactions(rxn_list)

    gpr.fit()

    K = gpr.Kcov_
    y = gpr.y_mol_
    alpha = gpr.alpha_mol_
    y_pred = K.dot(alpha)
    dy = y_pred - y
    rtkd = np.sqrt(np.diag(K))
    rxn_id_arr = np.array(rxn_id_list)
    nitems = 20
    for i, rxn_id in enumerate(rxn_id_list):
        rel_cov = K[i] / (rtkd * rtkd[i] + 1e-16)
        print(rxn_id)
        inds = np.argsort(np.abs(rel_cov))
        inds = np.flip(inds)[:nitems]
        print(rxn_id_arr[inds])
        print(rel_cov[inds])
        print(dy[inds])
        print()
    # Compute raw GP outputs for each molecule
    mol_raw_outputs = {}
    if mapping_plans and all(mfunc is not None for mfunc in mapping_plans):
        mapped_xc = gpr.map(mapping_plans)
        
        for dset_name in datasets_list:
            ddirs = find_datasets(dset_name, args, data_settings)
            with open(os.path.join(ddirs["SL"],
                                f"{dset_name}_settings.yaml"), "r") as f:
                mol_ids = yaml.load(f, Loader=yaml.CLoader)["MOL_IDS"]
            
            for mol_id in mol_ids:
                try:
                    # Read stored features and integration weights
                    data = MOLGP.load_data(ddirs, mol_id, get_orb_deriv=False)
                    desc = data["desc"]        # shape (nspin, nfeat, Nsamp)
                    weights = data["wt"]       # shape (Nsamp,)
                    
                    # Features are already normalized in the loaded data
                    X0T_norm = desc
                    
                    # Evaluate model and get raw GP output (baseline NOT applied)
                    _, _, f_raw, _ = mapped_xc(X0T_norm,
                                             rhocut=0,
                                             return_raw_ml_output=True)
                    
                    # Handle individual kernel outputs
                    if len(gpr.kernels) > 1:
                        # f_raw has shape (num_kernels, ...), process each kernel separately
                        kernel_outputs = []
                        for k_idx in range(len(gpr.kernels)):
                            f_k = f_raw[k_idx]  # Extract this kernel's output
                            
                            if f_k.ndim == 2:  # SEP mode
                                f_avg_k = (f_k * weights).sum(axis=1) / weights.sum()
                                kernel_outputs.append({
                                    "kernel_index": k_idx,
                                    "f_avg_per_spin": f_avg_k.tolist(),
                                    "f_avg_total": f_avg_k.sum(),
                                    "mode": "SEP"
                                })
                            else:  # NPOL/POL mode
                                f_avg_k = (f_k * weights).sum() / weights.sum()
                                kernel_outputs.append({
                                    "kernel_index": k_idx,
                                    "f_avg": float(f_avg_k),
                                    "mode": "NPOL/POL"
                                })
                        
                        # Also compute total (summed) output
                        f_raw_total = f_raw.sum(axis=0)
                        if f_raw_total.ndim == 2:
                            f_avg_total = (f_raw_total * weights).sum(axis=1) / weights.sum()
                            total_output = {
                                "f_avg_per_spin": f_avg_total.tolist(),
                                "f_avg_total": f_avg_total.sum(),
                                "mode": "SEP"
                            }
                        else:
                            f_avg_total = (f_raw_total * weights).sum() / weights.sum()
                            total_output = {
                                "f_avg": float(f_avg_total),
                                "mode": "NPOL/POL"
                            }
                        
                        mol_raw_outputs[mol_id] = {
                            "individual_kernels": kernel_outputs,
                            "total_output": total_output,
                            "num_kernels": len(gpr.kernels)
                        }
                    else:
                        # Single kernel case
                        if f_raw.ndim == 2:  # SEP mode: average each spin separately
                            f_avg = (f_raw * weights).sum(axis=1) / weights.sum()
                            mol_raw_outputs[mol_id] = {
                                "f_avg_per_spin": f_avg.tolist(),
                                "f_avg_total": f_avg.sum(),
                                "mode": "SEP",
                                "num_kernels": 1
                            }
                        else:  # NPOL/POL mode
                            f_avg = (f_raw * weights).sum() / weights.sum()
                            mol_raw_outputs[mol_id] = {
                                "f_avg": float(f_avg),
                                "mode": "NPOL/POL",
                                "num_kernels": 1
                            }
                    
                    print(f"{mol_id}: raw GP output avg = {mol_raw_outputs[mol_id]}")
                    
                except Exception as e:
                    print(f"Warning: Could not compute raw output for {mol_id}: {e}")
                    mol_raw_outputs[mol_id] = {"error": str(e)}

    ana_set = {
        "K": K,
        "Kfull": gpr.K_,
        "y_pred": y_pred,
        "y": y,
        "alpha": alpha,
        "rxn_id_list": rxn_id_list,
        "mol_raw_outputs": mol_raw_outputs,
    }
    with open("train_analysis.yaml", "w") as f:
        yaml.dump(ana_set, f, Dumper=yaml.CDumper)

    # Process validation dataset if provided
    if args.validation_dataset_file is not None:
        print("\nProcessing validation dataset...")
        with open(args.validation_dataset_file, "r") as f:
            val_data_settings = yaml.load(f, Loader=yaml.CLoader)
        
        # Store original reaction lists to restore later
        orig_rxn_ref_list = gpr.rxn_ref_list.copy()
        orig_rxn_noise_list = gpr.rxn_noise_list.copy()
        orig_rxn_cov_lists = []
        for kernel in gpr.kernels:
            orig_rxn_cov_lists.append(kernel.rxn_cov_list.copy())
        
        # Load validation datasets
        val_datasets_list = list(val_data_settings["systems"].keys())
        val_molid_map = {}
        
        # Create temporary args with validation feature names for find_datasets
        val_args = type(args)()
        for attr in dir(args):
            if not attr.startswith('_'):
                setattr(val_args, attr, getattr(args, attr))
        
        # Override with validation-specific feature names
        val_args.ref_feat_name = args.val_ref_feat_name or args.ref_feat_name
        val_args.sl_feat_name = args.val_sl_feat_name or args.sl_feat_name
        val_args.nldf_feat_name = args.val_nldf_feat_name or args.nldf_feat_name
        val_args.nlof_feat_name = args.val_nlof_feat_name or args.nlof_feat_name
        val_args.sdmx_feat_name = args.val_sdmx_feat_name or args.sdmx_feat_name
        val_args.hyb_feat_name = args.val_hyb_feat_name or args.hyb_feat_name
        
        # Store molecular covariances for validation data
        for val_dset_name in val_datasets_list:
            val_dirnames = find_datasets(val_dset_name, val_args, val_data_settings)
            with open(
                os.path.join(val_dirnames["SL"], "{}_settings.yaml".format(val_dset_name)), "r"
            ) as f:
                val_settings = yaml.load(f, Loader=yaml.CLoader)
                val_mol_ids = val_settings["MOL_IDS"]
            val_molid_map[val_dset_name] = val_mol_ids
            
            # Store covariances for validation molecules
            load_orbs = val_data_settings["systems"][val_dset_name].get("load_orbs")
            gpr.store_mol_covs(
                val_dirnames, val_mol_ids, get_orb_deriv=load_orbs, get_correlation=True
            )
        
        # Load validation reactions
        val_rxn_list = []
        val_rxn_id_list = []
        val_rxn_ids = list(val_data_settings["reactions"].keys())
        for i, rxn_id in enumerate(val_rxn_ids):
            rxn_dict = load_rxns(rxn_id)
            rxn_settings = val_data_settings["reactions"][rxn_id]
            mode = rxn_settings.get("mode") or 0
            for k, v in list(rxn_dict.items()):
                v.update(rxn_settings)
                val_rxn_id_list.append(k)
                val_rxn_list.append((mode, v))
        
        # Add validation reactions (this updates gpr's internal lists)
        gpr.add_reactions(val_rxn_list)
        
        # Write validation analysis
        write_validation_analysis(
            gpr, val_rxn_list, val_rxn_id_list, val_molid_map, 
            args, val_data_settings, mapping_plans, val_args=val_args
        )
        
        # Restore original reaction lists (remove validation data)
        gpr.rxn_ref_list = orig_rxn_ref_list
        gpr.rxn_noise_list = orig_rxn_noise_list
        for i, kernel in enumerate(gpr.kernels):
            kernel.rxn_cov_list = orig_rxn_cov_lists[i]
        
        print("Validation analysis written to validation_analysis.yaml")

    dump(gpr, args.save_file)

    if args.mapped_fname is not None:
        for mfunc in mapping_plans:
            assert mfunc is not None
        dump(gpr.map(mapping_plans), args.mapped_fname)


if __name__ == "__main__":
    main()
