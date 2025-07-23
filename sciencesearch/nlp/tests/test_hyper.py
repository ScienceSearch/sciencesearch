"""
Test of hyper module
"""

import pytest
from sciencesearch.nlp.hyper import Hyper
from .test_sweep import sweep_text, kpminer_sweep, yake_sweep, rake_sweep


@pytest.mark.unit
def test_hyper_init():
    Hyper()


@pytest.mark.unit
def test_hyper_get_top(kpminer_sweep, rake_sweep, yake_sweep):
    hyper = Hyper()
    hyper.add_sweep(kpminer_sweep)
    hyper.add_sweep(rake_sweep)
    hyper.add_sweep(yake_sweep)
    eps = 0.05
    true_keywords = ["gregor samsa", "vermin", "travelling salesman", "troubled dreams"]
    top = hyper.get_top_f1(sweep_text, true_keywords, epsilon=eps)
    assert len(top) > 0
    best = top[0].f1_score
    for res in top:
        score = res.f1_score
        assert best - score <= eps
