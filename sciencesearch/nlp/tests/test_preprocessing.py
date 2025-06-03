"""
Test cases for for the nlp.preprocessing module.
"""

import json
import logging
import os
import pytest
from sciencesearch.nlp.preprocessing import Preprocessor


@pytest.fixture
def log():
    log = logging.getLogger("sciencesearch.nlp.preprocessing")
    if os.environ.get("TEST_DEBUG", None):
        log.setLevel(logging.DEBUG)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        log.addHandler(h)


@pytest.fixture
def pp_text(request):
    pre_text_dir = request.path.parent / "test_files" / "preprocessing"
    for name in pre_text_dir.glob("*.json"):
        with (pre_text_dir / name).open("r") as f:
            yield json.load(f)


@pytest.mark.unit
def test_clean_individual_features(pp_text, log):
    prep = Preprocessor()
    for tc in pp_text:
        err_msg = f"Failed: {tc['description']}"
        assert (
            prep.process_string(text=tc["input_text"]) == tc["expected_output"]
        ), err_msg
