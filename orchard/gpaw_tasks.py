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

import copy
import os
import shlex
import shutil
import subprocess
import sys
import time

import yaml
from ase import Atoms
from fireworks import FiretaskBase, Firework, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize

from orchard.workflow_utils import get_save_dir

GPAW_CALL_SCRIPT = __file__.replace("gpaw_tasks", "gpaw_caller")
GPAW_DATA_SCRIPT = __file__.replace("gpaw_tasks", "gpaw_data_caller")

DEFAULT_GPAW_CALC_SETTINGS = {
    "xc": "PBE",
    "txt": "calc.txt",
    "maxiter": 200,
    "verbose": True,
    "spinpol": False,
    "kpts": (1, 1, 1),
    "hund": False,
}
DEFAULT_GPAW_CONTROL_SETTINGS = {
    "save_calc": False,
    "mode": 1000.0,
    "cider": None,
}


def setup_gpaw_cmd(struct, settings_inp, nproc=None, cmd=None, update_only=False):
    if nproc is None:
        if os.environ.get("NPROC_GPAW") is None:
            nproc = 1
        else:
            nproc = os.environ["NPROC_GPAW"]
    if cmd is not None:
        pass
    elif nproc == 1:
        cmd = "python {call_script} {settings_path}"
    else:
        cmd = "mpirun -np {nproc} python {call_script} {settings_path}"

    # logfile = settings_inp['calc'].get('txt') or DEFAULT_GPAW_CALC_SETTINGS.get('txt')
    # print('LOGFILE', logfile)
    # if logfile is not None:
    #    logfile = os.path.abspath(logfile)
    #    cmd = cmd + ' | tee {}'.format(logfile)

    if update_only:
        settings = {"calc": {}, "control": {"save_calc": False}}
    else:
        settings = {
            "calc": copy.deepcopy(DEFAULT_GPAW_CALC_SETTINGS),
            "control": copy.deepcopy(DEFAULT_GPAW_CONTROL_SETTINGS),
        }
    if "calc" in settings_inp.keys():
        settings["calc"].update(settings_inp["calc"])
    if "control" in settings_inp.keys():
        settings["control"].update(settings_inp["control"])
    settings_path = os.path.abspath("./gpaw_settings_tmp.yaml")
    if settings["control"]["save_calc"]:
        settings["control"]["save_calc"] = os.path.abspath("./gpaw_output_tmp.gpw")
    else:
        settings["control"]["save_calc"] = None

    settings["calc"][
        "txt"
    ] = "-"  # TODO should have nicer output settings at some point
    if update_only:
        assert isinstance(struct, str)
        settings["restart_file"] = struct
    elif isinstance(struct, dict):
        pass
    elif isinstance(struct, Atoms):
        struct = struct.todict()
    else:
        raise ValueError("struct must be dict or Atoms")
    settings["struct"] = struct

    with open(settings_path, "w") as f:
        yaml.dump(settings, f)
    cmd = cmd.format(
        nproc=nproc,
        call_script=GPAW_CALL_SCRIPT,
        settings_path=settings_path,
    )
    return cmd, settings["control"]["save_calc"], settings


def call_gpaw(cmd, logfile, require_converged=True):
    if logfile == "-":
        logfile = "calc.txt"
    logfile = os.path.abspath(logfile)
    f = open(logfile, "w")
    print("LOGFILE", logfile)
    start_time = time.monotonic()
    proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=f, stderr=f)
    proc.wait()
    stop_time = time.monotonic()
    f.close()
    if proc.returncode != 0:
        successful = False
        update_spec = {}
    else:
        with open("gpaw_outdata.tmp", "r") as f:
            results = yaml.load(f, Loader=yaml.Loader)
        if (not results["converged"]) and require_converged:
            successful = (
                False  # raise RuntimeError('GPAW calculation did not converge!')
            )
        else:
            successful = True
        # update_spec = {
        #    'e_tot': results['e_tot'],
        #    'converged': results['converged'],
        # }
        update_spec = results
    return successful, update_spec, stop_time - start_time, logfile


@explicit_serialize
class GPAWSinglePointSCF(FiretaskBase):

    required_params = ["struct", "settings", "method_name", "system_id"]
    optional_params = ["require_converged", "method_description", "nproc", "cmd"]

    def run_task(self, fw_spec):
        if self.get("require_converged") is None:
            self["require_converged"] = True
        cmd, save_file, settings = setup_gpaw_cmd(
            self["struct"],
            self["settings"],
            nproc=self.get("nproc"),
            cmd=self.get("cmd"),
            update_only=False,
        )

        logfile = settings["calc"].get("txt") or "calc.txt"
        successful, update_spec, wall_time, logfile = call_gpaw(
            cmd, logfile, require_converged=self["require_converged"]
        )
        struct = update_spec.get("struct") or self["struct"]

        update_spec.update(
            {
                "successful": successful,
                "logfile": logfile,
                "method_name": self["method_name"],
                "method_description": self.get("method_description"),
                "save_file": save_file,
                "settings": settings,
                "struct": struct,
                "system_id": self["system_id"],
                "wall_time": wall_time,
            }
        )
        return FWAction(update_spec=update_spec)


@explicit_serialize
class GPAWSinglePointRestart(FiretaskBase):

    required_params = ["new_settings", "new_method_name", "restart_file", "system_id"]
    optional_params = ["require_converged", "new_method_description", "nproc", "cmd"]

    def run_task(self, fw_spec):
        if self.get("require_converged") is None:
            self["require_converged"] = True
        cmd, save_file, settings = setup_gpaw_cmd(
            self["restart_file"],
            self["new_settings"],
            nproc=self.get("nproc"),
            cmd=self.get("cmd"),
            update_only=True,
        )
        run_fname = os.path.join(os.path.dirname(self["restart_file"]), "run_info.yaml")
        with open(run_fname, "r") as f:
            struct = yaml.load(f, Loader=yaml.Loader)["struct"]

        logfile = settings["calc"].get("txt") or "calc.txt"
        successful, update_spec, wall_time, logfile = call_gpaw(cmd, logfile)
        struct = update_spec.get("struct") or struct

        update_spec.update(
            {
                "successful": successful,
                "logfile": logfile,
                "method_name": self["new_method_name"],
                "method_description": self.get("new_method_description"),
                "save_file": save_file,
                "settings": settings,
                "struct": struct,
                "system_id": self["system_id"],
                "wall_time": wall_time,
            }
        )
        return FWAction(update_spec=update_spec)


@explicit_serialize
class SaveGPAWResults(FiretaskBase):

    required_params = ["save_root_dir"]
    optional_params = ["no_overwrite"]

    def run_task(self, fw_spec):
        save_dir = get_save_dir(
            self["save_root_dir"],
            "PW-KS",
            "",
            fw_spec["system_id"],
            functional=fw_spec["method_name"],
        )
        if self.get("no_overwrite"):
            exist_ok = False
        else:
            exist_ok = True
        os.makedirs(save_dir, exist_ok=exist_ok)

        if not fw_spec["successful"]:
            shutil.copyfile(fw_spec["logfile"], os.path.join(save_dir, "log.txt"))
            raise RuntimeError("GPAW job failed, see {}/log.txt".format(save_dir))

        out_data = {
            "struct": fw_spec["struct"],
            "settings": fw_spec["settings"],
            "e_tot": fw_spec["e_tot"],
            "converged": fw_spec["converged"],
            "wall_time": fw_spec["wall_time"],
            "method_description": fw_spec["method_description"],
        }
        out_file = os.path.join(save_dir, "run_info.yaml")
        with open(out_file, "w") as f:
            yaml.dump(out_data, f)

        if fw_spec["logfile"] is not None:
            shutil.copyfile(fw_spec["logfile"], os.path.join(save_dir, "log.txt"))
        if fw_spec["save_file"] is not None:
            shutil.copyfile(fw_spec["save_file"], os.path.join(save_dir, "calc.gpw"))

        return FWAction(stored_data={"save_dir": save_dir})


@explicit_serialize
class StoreFeatures(FiretaskBase):

    required_params = ["settings"]

    def run_task(self, fw_spec):
        if self["settings"].get("nproc") is not None:
            nproc = self["settings"].get("nproc")
        elif os.environ.get("NPROC_GPAW") is None:
            nproc = 1
        else:
            nproc = os.environ["NPROC_GPAW"]
        if nproc == 1:
            cmd = "python -u {call_script} {settings_path}"
        else:
            cmd = "mpirun -np {nproc} python -u {call_script} {settings_path}"

        print("NPROC", nproc)

        settings_path = os.path.abspath("./gpaw_settings_tmp.yaml")
        with open(settings_path, "w") as f:
            yaml.dump(self["settings"], f)
        cmd = cmd.format(
            nproc=nproc,
            call_script=GPAW_DATA_SCRIPT,
            settings_path=settings_path,
        )

        logfile = os.path.abspath("calc.txt")
        # with open(logfile, 'w') as f:
        #    print('LOGFILE', logfile)
        #    start_time = time.monotonic()
        #    #proc = subprocess.Popen(shlex.split(cmd), shell=False, stdout=f, stderr=f)
        #    proc = subprocess.Popen(shlex.split(cmd), shell=False, capture_output=True)
        #    return_code = proc.wait()
        #    assert return_code == 0
        #    stop_time = time.monotonic()
        #    print('Script runtime is {} s'.format(stop_time - start_time))

        print("LOGFILE", logfile)
        start_time = time.monotonic()
        proc = subprocess.Popen(
            shlex.split(cmd), shell=False, stdout=sys.stdout, stderr=sys.stderr
        )
        return_code = proc.wait()
        assert return_code == 0
        stop_time = time.monotonic()
        print("Script runtime is {} s".format(stop_time - start_time))


def make_etot_firework(
    struct,
    settings,
    method_name,
    system_id,
    save_root_dir,
    no_overwrite=False,
    require_converged=True,
    method_description=None,
    nproc=None,
    cmd=None,
    name=None,
):
    struct = struct.todict()
    t1 = GPAWSinglePointSCF(
        struct=struct,
        settings=settings,
        method_name=method_name,
        system_id=system_id,
        require_converged=require_converged,
        method_description=method_description,
        nproc=nproc,
        cmd=cmd,
    )
    t2 = SaveGPAWResults(save_root_dir=save_root_dir, no_overwrite=no_overwrite)
    return Firework([t1, t2], name=name)


def make_etot_firework_restart(
    new_settings,
    new_method_name,
    system_id,
    old_method_name,
    save_root_dir,
    no_overwrite=False,
    require_converged=True,
    new_method_description=None,
    nproc=None,
    cmd=None,
    name=None,
):
    restart_file = os.path.join(
        get_save_dir(
            save_root_dir,
            "PW-KS",
            "",
            system_id,
            functional=old_method_name,
        ),
        "calc.gpw",
    )
    t1 = GPAWSinglePointRestart(
        new_settings=new_settings,
        new_method_name=new_method_name,
        restart_file=restart_file,
        system_id=system_id,
        require_converged=require_converged,
        new_method_description=new_method_description,
        nproc=nproc,
        cmd=cmd,
    )
    t2 = SaveGPAWResults(save_root_dir=save_root_dir, no_overwrite=no_overwrite)
    return Firework([t1, t2], name=name)
