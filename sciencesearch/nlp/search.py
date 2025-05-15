"""
Demonstration of a search built from config file
"""

# stdlib
from collections import defaultdict
import csv
from glob import glob
import importlib
import json
import logging
from pathlib import Path

# package
from .models import Ensemble
from .sweep import Sweep
from .hyper import Hyper
from .train import load_hyper, run_hyper, train_hyper

logging.root.setLevel(logging.ERROR)  # quiet pke warnings


class Searcher:
    """Dead simple search that can be created from a config file and some input files."""

    def __init__(self, file_keywords=None):
        """Constructor."""
        self._db = defaultdict(set)
        self._fkw = {}
        if file_keywords:
            self.add_entries(file_keywords)

    def add_entries(self, file_keywords: dict[str, list[str]]):
        """Add a map of files to a list of keywords."""
        self._fkw.update(file_keywords)
        for fname, kwlist in file_keywords.items():
            for kw in kwlist:
                self._db[kw].add(fname)

    @property
    def file_keywords(self) -> dict[str, list[str]]:
        """ "Get the current mapping of files to the list of keywords associated with each."""
        return self._fkw.copy()

    @classmethod
    def from_config(cls, config_file) -> "Searcher":
        conf = json.load(open(config_file))
        training = conf["training"]
        file_dir = Path(training["directory"])
        save_file = file_dir / training["save_file"]
        use_saved_results = False
        if save_file.exists():
            # TODO: check that it's newer than conf/input files
            use_saved_results = True
        if not use_saved_results:
            # load algorithms
            alg = {}
            for alg_name, entry in conf["algorithms"].items():
                m = importlib.import_module(entry["module"])
                alg[alg_name] = getattr(m, entry["class"])
            # set up hyperparameter sweeps
            hyper = Hyper()
            sweeps = conf["sweeps"]
            for alg_name, params in sweeps.items():
                swp = Sweep(alg[alg_name])
                for param_name, param_val in params.items():
                    if param_name[0] == "-":
                        continue  # skip, like a comment
                    ptype = param_val["_type"]
                    del param_val["_type"]
                    if ptype == "range":
                        swp.set_param_range(param_name, **param_val)
                    elif ptype == "discrete":
                        swp.set_param_discrete(param_name, param_val["values"])
                hyper.add_sweep(swp)
            # load training data
            file_keywords = {}
            for kwd_file in training["keywords"]:
                with open(file_dir / kwd_file) as f:
                    rdr = csv.reader(f)
                    for row in rdr:
                        filename, kw_str = row
                        keywords = [k.strip() for k in kw_str.split(",")]
                        file_keywords[filename] = keywords
            hyper_results = train_hyper(
                hyper,
                file_keywords,
                epsilon=training["epsilon"],
                save_file=save_file,
                directory=file_dir,
            )
        else:
            hyper_results = load_hyper(save_file)
        # run 'best' algorithm on input files
        search_kw = {}
        for input_file_pat in training["input_files"]:
            for fname in glob(input_file_pat, root_dir=file_dir):
                kw = run_hyper(hyper_results, file_dir / fname)
                search_kw[fname] = kw
        # initialize this class with results
        return cls(search_kw)

    def find(self, *keywords):
        files = set()
        for k in keywords:
            for fname in self._db[k]:
                files.add(fname)
        return files
