from sciencesearch.nlp.export import *
import json
import pytest


@pytest.mark.unit
def test_excel_export(tmp_path):
    p = tmp_path / "excel_export.xls"
    # write out
    with p.open("wb") as f:
        with ExcelExporter(f) as e:
            e.header(("one", "two", "three"))
            e.row((1, 2, 3))
            e.row(("a", "b", "c"))
    # read in
    with p.open("rb") as f:
        df = pd.read_excel(f)
    # check
    print(df)
    assert df.iloc[0, 0] == 1
    assert df.iloc[1, 0] == "a"


@pytest.mark.unit
def test_json_export(tmp_path):
    p = tmp_path / "json_export.json"
    # write out
    with p.open("w") as f:
        with JsonExporter(f) as e:
            e.header(("one", "two", "three"))
            e.row((1, 2, 3))
            e.row(("a", "b", "c"))
    # read in
    with p.open("r") as f:
        d = json.load(f)
    # check
    print(d)
    data = d["data"]
    assert data[0]["one"] == 1
    assert data[1]["one"] == "a"


@pytest.mark.unit
@pytest.mark.parametrize(
    "format_name,expected",
    [
        ("excel", ExcelExporter),
        ("Excel", ExcelExporter),
        (ExportFormat.EXCEL, ExcelExporter),
        ("json", JsonExporter),
        ("JSON", JsonExporter),
        (ExportFormat.JSON, JsonExporter),
        ("foo", None),
    ],
)
def test_get_exporter_class(format_name, expected):
    if expected is None:
        with pytest.raises(KeyError):
            get_exporter_class(format_name)
    else:
        assert get_exporter_class(format_name) == expected
