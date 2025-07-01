from sciencesearch.nlp.export import *
import json
import pytest


@pytest.mark.unit
def test_excel_export(tmp_path):
    p = tmp_path / "excel_export.xls"
    # write out
    with ExcelExporter(p) as e:
        e.header(("item", "one", "two", "three"))
        e.row(("foo", 1, 2, 3))
        e.row(("bar", "a", "b", "c"))
    # read in
    with p.open("rb") as f:
        df = pd.read_excel(f)
    # check
    print(df)
    assert df.iloc[0, 0] == "foo"
    assert df.iloc[0, 1] == 1
    assert df.iloc[1, 0] == "bar"
    assert df.iloc[1, 1] == "a"


@pytest.mark.unit
def test_json_export(tmp_path):
    p = tmp_path / "json_export.json"
    # write out
    with JsonExporter(
        p,
        {
            "foo": {"one": 1, "two": 2, "three": 3},
            "bar": {"one": "a", "two": "b", "three": "c"},
        },
    ) as e:
        pass
    # read in
    with p.open("r") as f:
        d = json.load(f)
    # check
    print(d)
    assert d["foo"]["one"] == 1
    assert d["foo"]["two"] == 2
    assert d["bar"]["one"] == "a"
    assert d["bar"]["two"] == "b"


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
@pytest.mark.unit
def test_get_exporter_class(format_name, expected):
    if expected is None:
        with pytest.raises(KeyError):
            get_exporter_class(format_name)
    else:
        assert get_exporter_class(format_name) == expected


@pytest.mark.unit
def test_export_func(tmp_path):
    data = {
        "file1": {"training": "a, b, c", "predicted": "d, e, f"},
        "file2": {"training": "a, b, c", "predicted": "x, e, z"},
        "file3": {"training": "a, b, c", "predicted": "p, e, r"},
    }
    p = tmp_path / "export_func.json"
    export(data, output_filename=p, output_format="json")
    data2 = json.load(p.open("r"))
    assert data == data2
