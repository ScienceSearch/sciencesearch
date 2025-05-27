import json
import pytest
from sciencesearch.nlp.preprocessing import clean_text


def read_test_cases_from_json(file_path):
    test_cases = []

    with open(file_path, "r") as file:
        data = json.load(file)

    for item in data:
        test_cases.append(
            (item["description"], item["input_text"], item["expected_output"])
        )

    return test_cases


test_cases = read_test_cases_from_json(
    "sciencesearch/nlp/tests/test_files/preprocessing/preprocessing_test_text.json"
)


@pytest.mark.parametrize("description, input_text, expected_output", test_cases)
def test_clean_individual_features(description, input_text, expected_output):
    cleaned_text = clean_text(text=input_text)
    assert (
        cleaned_text == expected_output
    ), f"Cleaned text does not match expected output for case {description}"