# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
from typing import Union, List
import contextlib
import logging
import os
import urllib
from pathlib import Path
import re

from tqdm import tqdm

logger = logging.getLogger(__name__)


# Define a context manager to temporarily mute print statements
@contextlib.contextmanager
def mute_print():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    import logging

    kwargs = snakemake.config.get("logging", dict()).copy()
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath("..", "logs", f"{snakemake.rule}.log")
        logfile = snakemake.log.get("python", snakemake.log[0] if snakemake.log else fallback_path)
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ]
            }
        )
    logging.basicConfig(**kwargs)


def progress_retrieve(url, file, disable=False):
    if disable:
        urllib.request.urlretrieve(url, file)
    else:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:

            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update(b * bsize - t.n)

            urllib.request.urlretrieve(url, file, reporthook=update_to)


class Dict(dict):
    """
    Dict is a subclass of dict, which allows you to get AND SET items in the
    dict using the attribute syntax!

    Stripped down from addict https://github.com/mewwts/addict/ .
    """

    def __setattr__(self, name, value):
        """
        setattr is called when the syntax a.b = 2 is used to set a value.
        """
        if hasattr(Dict, name):
            raise AttributeError("'Dict' object attribute " "'{0}' is read-only".format(name))
        self[name] = value

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError as e:
            raise AttributeError(e.args[0])

    def __delattr__(self, name):
        """
        Is invoked when del some_addict.b is called.
        """
        del self[name]

    _re_pattern = re.compile("[a-zA-Z_][a-zA-Z0-9_]*")

    def __dir__(self):
        """
        Return a list of object attributes.

        This includes key names of any dict entries, filtered to the
        subset of valid attribute names (e.g. alphanumeric strings
        beginning with a letter or underscore).  Also includes
        attributes of parent dict class.
        """
        dict_keys = []
        for k in self.keys():
            if isinstance(k, str):
                m = self._re_pattern.match(k)
                if m:
                    dict_keys.append(m.string)

        obj_attrs = list(dir(Dict))

        return dict_keys + obj_attrs


def mock_snakemake(rulename: str, configfiles: Union[List[str], str] = [], **wildcards):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    configfiles: list, str
        list of configfiles to be used to update the config
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from packaging.version import Version, parse

    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent

    user_in_script_dir = Path.cwd().resolve() == script_dir
    if user_in_script_dir:
        os.chdir(root_dir)
    elif Path.cwd().resolve() != root_dir:
        raise RuntimeError(
            "mock_snakemake has to be run from the repository root" f" {root_dir} or scripts directory {script_dir}"
        )
    try:
        for p in sm.SNAKEFILE_CHOICES:
            if os.path.exists(p):
                snakefile = p
                break
        kwargs = dict(rerun_triggers=[]) if parse(sm.__version__) > Version("7.7.0") else {}
        if isinstance(configfiles, str):
            configfiles = [configfiles]

        workflow = sm.Workflow(snakefile, overwrite_configfiles=configfiles, **kwargs)
        workflow.include(snakefile)

        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
                workflow.configfile(f)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = Dict(wildcards)
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessable(*ios):
            for io in ios:
                for i in range(len(io)):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    finally:
        if user_in_script_dir:
            os.chdir(script_dir)
    return snakemake

def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    import logging

    kwargs = snakemake.config.get("logging", dict()).copy()
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath(
            "..", "logs", f"{snakemake.rule}.log"
        )
        logfile = snakemake.log.get(
            "python", snakemake.log[0] if snakemake.log else fallback_path
        )
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ]
            }
        )
    logging.basicConfig(**kwargs)