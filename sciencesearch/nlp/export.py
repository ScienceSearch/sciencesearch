"""
Export the results of an analysis.
"""

from abc import ABC, abstractmethod
from enum import Enum
import json
from typing import List, Any, Union

import pandas as pd


class ExportFormat(Enum):
    EXCEL = "excel"
    JSON = "json"


class Exporter(ABC):
    """Base class for exporters."""

    def __init__(self, output_file):
        self._of = output_file

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
    def header(self, data: List[Any]):
        self._fields = data
        self._d = {"data": []}
        self._data = self._d["data"]

    def row(self, data: List[Any]):
        obj = {self._fields[i]: data[i] for i in range(len(self._fields))}
        self._data.append(obj)

    def close(self):
        json.dump(self._d, self._of)


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
