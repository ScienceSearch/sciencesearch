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
from IPython.core.display import HTML

# package
from .sweep import Sweep
from .hyper import Hyper
from .train import load_hyper, run_hyper, train_hyper
from .visualize_kws import JsonView

logging.root.setLevel(logging.ERROR)  # quiet pke warnings


class Searcher:
    """Dead simple search that can be created from a config file and some input files."""

    def __init__(
        self,
        predicted_keywords: dict[str, list[str]] = None,
        training_keywords: dict[str, list[str]] = None,
        file_keywords: dict[str, list[str]] = None,
        config_file: str = None,
    ):
        """Constructor."""
        self.config_file = config_file
        self._db = defaultdict(set)
        self._fkw = {}
        self._pred_kws = predicted_keywords or {}
        self._training_kws = training_keywords or {}

        if predicted_keywords:
            self.add_entries(predicted_keywords)
        if training_keywords:
            self.add_entries(training_keywords)
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

    @property
    def predicted_keywords(self) -> dict[str, list[str]]:
        """ "Get the current mapping of files to the list of keywords associated with each."""
        return self._pred_kws.copy()

    @property
    def training_keywords(self) -> dict[str, list[str]]:
        """ "Get the current mapping of files to the list of keywords associated with each."""
        return self._training_kws.copy()

    def training_and_predicted_keywords(self) -> dict[str, dict[list[str]]]:
        """Saves all keyword sets in a combined JSON structure.

        Creates a JSON file where each filename maps to a dictionary containing
        both training and tuned (predicted) keyword sets.

        Args:
            filename (str): Path to the output JSON file.

        Note:
            The output structure is:
            {
                "filename1.txt": {
                    "training": ["keyword1", "keyword2", ...],
                    "tuned": ["keyword3", "keyword4", ...]
                },
                ...
            }
        """
        res = {}
        for fn, keywords in self.file_keywords.items():
            all_kws = {
                "training": self.training_keywords.get(fn, []),
                "tuned": self.predicted_keywords.get(fn, []),
            }
            res[fn] = all_kws
        return res

    @classmethod
    def from_config(cls, config_file) -> "Searcher":
        conf = json.load(open(config_file))
        training = conf["training"]
        file_dir = Path(training["directory"])
        save_file = file_dir / training["save_file"]
        use_saved_results = False
        file_keywords = {}
        for kwd_file in training["keywords"]:
            with open(file_dir / kwd_file) as f:
                rdr = csv.reader(f)
                for row in rdr:
                    filename, kw_str = row
                    keywords = [k.strip() for k in kw_str.split(",")]
                    file_keywords[filename] = keywords
        search_kw = {}
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
        for input_file_pat in training["input_files"]:
            for fname in glob(input_file_pat, root_dir=file_dir):
                kw = run_hyper(hyper_results, file_dir / fname)
                search_kw[fname] = kw
        # initialize this class with results
        return cls(
            predicted_keywords=search_kw,
            training_keywords=file_keywords,
            config_file=config_file,
        )

    def find(self, *keywords):
        files = set()
        for k in keywords:
            for fname in self._db[k]:
                files.add(fname) 
        return files

    def view_keywords(
        self,
        show_training: bool = False,
        show_predicted: bool = False,
        textfilename: str = None,
    ):
        data = {}
        if show_training and show_predicted:
            data = self.training_and_predicted_keywords()
        elif show_training:
            data = self.training_keywords
        elif show_predicted:
            data = self.predicted_keywords
        html = JsonView.visualize_from_config(
            config_file=self.config_file,
            data=data,
            save_filename="vis_kws",
            textfilename=textfilename,
        )
        return html
