"""
Perform a sweep with NLP models
"""

# stdlib
from collections import defaultdict
import itertools
from operator import attrgetter
from typing import Optional, Any, Iterable

# third-party
from sklearn.metrics import f1_score
import pandas as pd

# package
from sciencesearch.nlp.models import Algorithm


def _get_true_pred(
    true_kw: Iterable[str], pred_kw: Iterable[str]
) -> tuple[list[int], list[int]]:
    """Get true and predicted values given a list of
    'true' and 'predicted' keywords.

    Args:
        true_kw: True (ground truth) keywords
        pred_kw: Predicted (calculated) keywords

    Returns:
        A trio of lists, the first being a list of keywords including
        all unique values in the input, and the second two being a
        list of whether that keyword was positive (1) or negative (0)
        according to the 'true' or 'pred' list.
    """
    true_kw, pred_kw = set(true_kw), set(pred_kw)
    all_kw = true_kw.union(pred_kw)
    y_true, y_pred = [], []
    for kw in all_kw:
        y_true.append(1 if kw in true_kw else 0)
        y_pred.append(1 if kw in pred_kw else 0)
    return all_kw, y_true, y_pred


def get_f1_score(ground_truth: Iterable[str], extracted: Iterable[str]) -> float:
    all_kw, y_true, y_pred = _get_true_pred(ground_truth, extracted)
    return f1_score(y_true, y_pred)


########################################

F1_SCORE = "F1"


class ExtractionResult:

    def __init__(self, algorithm: str, parameters: dict[str, Any], keywords: list[str]):
        self.algorithm = algorithm
        self.parameters = parameters
        self.keywords = keywords
        self.scores = {}

    def add_score(self, name, value):
        self.scores[name] = value

    @property
    def f1_score(self):
        return self.scores[F1_SCORE]

    def get_dict(self):
        d = self.parameters.copy()
        d["keywords"] = self.keywords
        d.update(self.scores)
        return d


class SweepResult:

    def __init__(self):
        self._r = []
        self.ground_truth = []

    @property
    def results(self):
        return self._r.copy()

    def add_result(self, ext: ExtractionResult):
        self._r.append(ext)

    def add_score(self, score_name: str, calculate_score=None):
        if not self.ground_truth:
            raise ValueError(
                "Set attribute 'ground_truth' to a list of keywords before calling add_score()"
            )
        for r in self._r:
            score = calculate_score(self.ground_truth, r.keywords)
            r.add_score(score_name, score)

    def add_f1_score(self):
        return self.add_score(score_name=F1_SCORE, calculate_score=get_f1_score)

    def as_dataframe(self) -> pd.DataFrame:
        data = [x.get_dict() for x in self._r]
        return pd.DataFrame(data)

    def get_all_keywords(self) -> set[str]:
        all_kw = set()
        for r in self._r:
            all_kw = all_kw.union(set(r.keywords))
        return all_kw


class Sweep:
    """Sweep with a given algorithm and set of parameters."""

    def __init__(self, alg: type, **params):
        try:
            assert issubclass(alg, Algorithm)
        except (AssertionError, TypeError):
            raise TypeError("Input must be a subclass (not instance) of Algorithm")
        self._alg = alg
        self._alg_name = alg.__name__
        self._alg_params = params
        self._ranges = {}

    def set_param_range(
        self,
        name: str,
        lb: float | int,
        ub: float | int,
        step: Optional[float | int] = None,
        nsteps: Optional[int] = None,
    ):
        """Set range of parameters

        For determining the intermediate values,
        `step` is checked first then `nsteps`. One of
        them must be set.

        Args:
            name: Parameter name
            lb: Range lower bound
            ub: Range upper boound
            step: Step. Defaults to None, in which case use `nsteps`.
            nsteps: Number of steps. Defaults to None, using `step`.

        Raises:
            KeyError: Unknown parameter name
            ValueError: nstep or nsteps must be provided
        """
        if name in self._alg_params:
            raise ValueError(f"Parameter '{name}' is already set to a fixed value")
        alg_params = {x.name for x in self._alg.PARAM_SPEC}
        if name not in alg_params:
            raise KeyError(f"Cannot set range of unknown parameter {name}")
        if not type(lb) == type(ub):
            raise TypeError("Lower and upper bounds must be the same type")
        is_int = isinstance(lb, int)
        delta = ub - lb
        if step is None:
            if nsteps is None:
                raise ValueError("Must set one of 'step' or 'nsteps'")
            if is_int:
                step = delta // nsteps
            else:
                step = delta / nsteps
        if is_int:
            # add one to `ub`` so range includes upper bound
            self._ranges[name] = list(range(lb, ub + 1, step))
        else:
            # include upper bound, if within epsilon
            ub_epsilon = ub + min(delta / 1e3, 1e-06)
            r, i = [], 0
            while True:
                v = lb + i * step
                if v > ub_epsilon:
                    break
                r.append(v)
                i += 1
            self._ranges[name] = r

    def set_param_discrete(self, name: str, values: list):
        """Set a list of discrete parameter values.

        Args:
            name: Parameter name
            values: List of values to use
        """
        if name in self._alg_params:
            raise ValueError(f"Parameter '{name}' is already set to a fixed value")
        self._ranges[name] = values

    def run(self, text: str) -> SweepResult:
        """Run the sweep on the given text.

        Raises:
            ValueError: If no ranges specified

        Returns:
            List of results, with one result for each parameter combination
        """
        if not self._ranges:
            raise ValueError("No ranges specified")
        # order ranges so we can rebuild from tuples
        rnames = sorted(self._ranges.keys())
        rvalues = [self._ranges[k] for k in rnames]
        # go through all possible combinations
        sweep_result = SweepResult()
        for item in itertools.product(*rvalues):
            params = {rnames[i]: item[i] for i in range(len(item))}
            params.update(self._alg_params)
            alg = self._alg(**params)
            kw = alg.run(text)
            one_result = ExtractionResult(
                algorithm=self._alg_name, parameters=params.copy(), keywords=kw
            )
            sweep_result.add_result(one_result)
        return sweep_result


# TODO: test this!!
class Hyper:
    def __init__(self):
        self._sweeps = []

    def add_sweep(self, sweep: Sweep):
        self._sweeps.append(sweep)

    def get_top(
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
        while i >= 0 and ex_res[i].f1_score < lb:
            i -= 1
        return ex_res[i + 1 :]
