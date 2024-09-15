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
import traceback
import warnings
from argparse import ArgumentParser

import numpy as np
import yaml
from ciderpress.dft.baselines import BASELINE_CODES
from ciderpress.dft.settings import LDA_FACTOR, FeatureSettings
from ciderpress.dft.transform_data import FeatureList
from ciderpress.models.dft_kernel import DFTKernel
from ciderpress.models.train import MOLGP, strk_to_tuplek
from joblib import dump, load

from orchard.workflow_utils import load_rxns

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
        print(fname)
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
        assert os.path.exists(plan_file)
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
    for mol_id in mol_ids:
        data = MOLGP.load_data(dirnames, mol_id, None, "new")
        cond = data["desc"][:, 0, :] > args.density_cutoff
        print(data["desc"].shape, data["val"].shape)
        y = data["val"][cond] / (LDA_FACTOR * data["desc"][:, 0][cond] ** (4.0 / 3)) - 1
        cond = np.all(cond, axis=0)
        desc = data["desc"][:, :, cond]
        X = feat_settings.normalizers.get_normalized_feature_vector(desc)
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
            Xlist.append(X[..., ::n])
            if has_ddesc:
                GXOlist.append((ddesc[("O", 0)][0], GXO[..., ::n]))
                GXUlist.append((ddesc[("U", 0)][0], GXU[..., ::n]))
            ylist.append(y[::n])
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
        for dset_name in datasets_list:
            n = data_settings["systems"][dset_name].get("inverse_sampling_density")
            (
                Xlist_tmp,
                GXOlist_tmp,
                GXUlist_tmp,
                y_tmp,
                dset_ids,
            ) = parse_dataset_for_ctrl(dset_name, n, args, data_settings, settings)
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
            kernels.append(
                DFTKernel(
                    None,
                    feature_list,
                    plan["mode"],
                    BASELINE_CODES[plan["multiplicative_baseline"]],
                    additive_baseline=BASELINE_CODES.get(plan["additive_baseline"]),
                    ctrl_tol=ctrl_tol,
                    ctrl_nmax=ctrl_nmax,
                    component=plan.get("component"),
                )
            )
            if "lscale_override" in plan:
                lscale = np.array(plan.pop("lscale_override"))
                val_pca = None
                deriv_pca = None
            else:
                X1 = kernels[-1].X0Tlist_to_X1array(Xlist)
                # DXO1 = get_fd_x1(kernels[-1], GXRlist, GXOlist)
                # DXU1 = get_fd_x1(kernels[-1], GXRlist, GXUlist)
                # val_pca = analyze_cov(X1)
                # analyze_cov(DXO1, avg_and_std=val_pca[:2])
                # analyze_cov(DXU1, avg_and_std=val_pca[:2])
                # deriv_pca = analyze_cov(DXU1 - DXO1, avg_and_std=val_pca[:2])
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
                val_pca=val_pca,
                deriv_pca=deriv_pca,
            )
            kernels[-1].set_kernel(kernel)
            if "mapping_plan" in dir(plan_module):
                mfunc = plan_module.mapping_plan
            else:
                mfunc = None
            mapping_plans.append(mfunc)

        gpr = MOLGP(
            kernels,
            settings,
            libxc_baseline=args.libxc_baseline,
            default_noise=args.mol_sigma,
        )
        gpr.args = args

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
    ana_set = {
        "K": K,
        "Kfull": gpr.K_,
        "y_pred": y_pred,
        "y": y,
        "alpha": alpha,
        "rxn_id_list": rxn_id_list,
    }
    with open("train_analysis.yaml", "w") as f:
        yaml.dump(ana_set, f, Dumper=yaml.CDumper)

    dump(gpr, args.save_file)

    if args.mapped_fname is not None:
        for mfunc in mapping_plans:
            assert mfunc is not None
        dump(gpr.map(mapping_plans), args.mapped_fname)


if __name__ == "__main__":
    main()
