"""
Hyperparameter tuning
"""

# stdlib
from operator import attrgetter

# package
from sciencesearch.nlp.models import Algorithm
from sciencesearch.nlp.sweep import ExtractionResult, Sweep


class Hyper:
    def __init__(self):
        self._sweeps = []

    def add_sweep(self, sweep: Sweep):
        self._sweeps.append(sweep)

    def get_top_f1(
        self, text: str, keywords: list[str], epsilon=0.1
    ) -> list[ExtractionResult]:
        """Return all parameter/alg combinations that resulted in an F1 score
        within 'epsilon' of the best one, on the given text with the given
        ground truth keywords.
        """
        ex_res = []
        for sw in self._sweeps:
            res = sw.run(text)
            res.ground_truth = keywords
            res.add_f1_score()
            # add to flattened list of ExtractionResult objects
            ex_res.extend(res.results)
        # sort all results by F1 score
        ex_res.sort(key=attrgetter("f1_score"))
        # return top results within 'epsilon'
        i = len(ex_res) - 1
        top = ex_res[i].f1_score
        lb = top - epsilon
        while i >= 0 and ex_res[i].f1_score >= lb:
            i -= 1
        return ex_res[i + 1 :]


def algorithms_from_results(er: list[ExtractionResult]) -> list[Algorithm]:
    obj_list = []
    unique_alg = set()
    for res in er:
        # build unique key for algorithm and parameters
        key = [id(res.algorithm)]
        for pkey, pval in res.parameters.items():
            key.append((pkey, str(pval)))
        if key in unique_alg:
            continue
        unique_alg.add(key)
        obj_list.append(res.algorithm(**res.parameters))
    return obj_list
