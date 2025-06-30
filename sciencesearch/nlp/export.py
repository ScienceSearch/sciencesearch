"""
Export the results of an analysis.
"""

from abc import ABC, abstractmethod
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Union, Optional

# third-party
import pandas as pd


class ExportFormat(Enum):
    EXCEL = "excel"
    JSON = "json"


class Exporter(ABC):
    """Base class for exporters."""

    is_binary = None  #: whether output is binary or text. Set True/False in subclasses

    def __init__(self, output_file, data=None):
        self._of = output_file
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
        self._of.close()

    @abstractmethod
    def header(self, data):
        pass

    @abstractmethod
    def row(self, data):
        pass

    @abstractmethod
    def close(self):
        pass


class ExcelExporter(Exporter):

    is_binary = True

    def header(self, data: List[Any]):
        self._col = {n: [] for n in data}
        self._colnames = data

    def row(self, data: List[Any]):
        for i, item in enumerate(data):
            self._col[self._colnames[i]].append(item)

    def close(self):
        df = pd.DataFrame(self._col)
        df.to_excel(self._of, index=False)


class JsonExporter(Exporter):

    is_binary = False

    def header(self, data: List[Any]):
        return

    def row(self, data: List[Any]):
        return

    def close(self):
        json.dump(self._data, self._of)


_exporters = {ExportFormat.EXCEL: ExcelExporter, ExportFormat.JSON: JsonExporter}


def get_exporter_class(fmt: Union[str, ExportFormat]) -> Exporter:
    """Get the subclass of Exporter that should be used for the given input format.

    Args:
        fmt: Input format, as a string ('excel') or an :class:`ExportFormat`.

    Returns:
        Subclass of :class:`Exporter`

    Raises:
        KeyError: If the export format is unknown, which could happen for a string value.
    """
    if isinstance(fmt, str):
        s = fmt.lower()
        try:
            fmt = ExportFormat(s)
        except ValueError:
            formats = ", ".join([f.value for f in list(ExportFormat)])
            raise KeyError(f"Unknown export format. Must be one of: {formats}")
    return _exporters[fmt]


def export(
    data: Dict[str, Dict[str, Any]],
    output_filename: Optional[Union[str, Path]] = None,
    output_format: ExportFormat | str = ExportFormat.EXCEL,
):
    """Export keywords (and optional context) to a file.

    Args:
        data: format = `{"item_name": {"value1_name": value1, ..}, ...}`
        output_filename: Output file name or path (if None, use stdout)
        output_format: Output format

    Raises:
        ValueError: Nothing to export
        ValueError: Unknown export format in `output_format`
    """
    if not data:
        raise ValueError("Nothing to export")
    try:
        exporter_class = get_exporter_class(output_format)
    except KeyError:
        raise ValueError("Cannot get exporter class")
    # initialize chosen exporter class
    if output_filename is None:
        output_file = sys.stdout
    else:
        out_p = Path(output_filename)
        if exporter_class.is_binary:
            output_file = out_p.open("wb")
        else:
            output_file = out_p.open("w")
    exporter = exporter_class(output_file, data=data)
    # export each row
    hdr = None
    for name, values in data.items():
        if hdr is None:
            hdr = list(values.keys())
            exporter.header(["name"] + hdr)
        row = [name] + [values[k] for k in hdr]
        exporter.row(row)
    # close it
    exporter.close()
