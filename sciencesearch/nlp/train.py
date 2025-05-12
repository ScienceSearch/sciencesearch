"""
Training pipeline
"""

from pathlib import Path
import pickle
from .hyper import Hyper
from .models import KPMiner, Rake, Yake, Ensemble


def train_hyper(
    hyperparameter: Hyper,
    file_kw: dict[str, list[str]],
    epsilon: float = 0.1,
    save_file: str | Path = None,
    directory: Path = None,
):
    """Train the hyperparameters on a list of text files with associated expected keywords.

    Args:
        hyperparameter: Hyperparameter class for running the training.
        file_kw: Mapping of filenames to a list of keywords to use as the gold_standard.
        epsilon: How close to the best F1 score a score must be to be considered 'best'
        save_file: Where to pickle the results. Defaults to None, meaning don't save
        directory:

    Raises:
        ValueError: _description_
    """
    hyper_results = []
    root_dir = directory if directory else Path.cwd()
    for training_file, gold_kw in file_kw.items():
        with open(root_dir / training_file) as f:
            print(f"Processing file: {Path(f.name).name}")
            text = f.read()
            res = hyperparameter.get_top_f1(text, gold_kw, epsilon=epsilon)
            hyper_results.extend(res)
    # save to file
    if save_file:
        with open(save_file, "wb") as f:
            pickle.dump(hyper_results, f)
    return hyper_results


def load_hyper(save_file: str | Path):
    with open(save_file, "rb") as f:
        hyper_results = pickle.load(f)
    return hyper_results


def run_hyper(
    hyper_results,
    text_file: str | Path = None,
    text: str = None,
    num_keywords: int = 10,
):
    # pick one from each algorithm
    alg_map, n = {KPMiner: None, Rake: None, Yake: None}, 0
    for res in hyper_results:
        if alg_map[res.algorithm] is None:
            alg_map[res.algorithm] = res.algorithm(**res.parameters)
            n += 1
        if n == 3:
            break
    ensm = Ensemble(
        *alg_map.values(),
        num_keywords=num_keywords,
        keyword_sort=[{"score": 0.75, "occ": 0.25}],
    )
    if text_file is not None:
        text = Path(text_file).open().read()
    elif text is None:
        raise ValueError("One of 'text' or 'text_file' should be provided")
    return ensm.run(text)
