"""
Export the results of an analysis.
"""

from abc import ABC, abstractmethod
from enum import Enum
from io import StringIO, BytesIO
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Union, Optional

# third-party
from graphviz import Graph
import pandas as pd


class ExportFormat(Enum):
    EXCEL = "excel"
    JSON = "json"
    GRAPHVIZ = "graph"


class Exporter(ABC):
    """Base class for exporters."""

    is_binary = None  #: whether output is binary or text. Set True/False in subclasses
    feed_rows = True  #: whether to feed each row to the exporter
    _open_file = True  #: whether constructor should open the file for writing

    # Special field names
    CONTENT = "content"
    NAME = "name"
    PREDICTED = "predicted"

    def __init__(self, output_filename, data=None):
        self._data = data
        self.as_string = False

        if self._open_file:
            if output_filename is None:
                output_file = BytesIO() if self.is_binary else StringIO()
                self.as_string = True
            elif output_filename == "-" or output_filename == "":
                if self.is_binary:
                    raise ValueError("Cannot write binary data to standard output")
                output_file = sys.stdout
            else:
                out_p = Path(output_filename)
                mode = "wb" if self.is_binary else "w"
                output_file = out_p.open(mode)
            self._of, self._path = output_file, None
        else:
            self._of = None
            if output_filename:
                self._path = Path(output_filename)
            else:
                self._path = None
                self.as_string = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()
        self._of.close()

    def header(self, data):
        pass

    def row(self, data):
        pass

    @abstractmethod
    def close(self):
        pass

    def getvalue(self) -> str:
        if self._of:
            return self._of.getvalue()
        return ""


class ExcelExporter(Exporter):

    is_binary = True
    feed_rows = True

    def header(self, data: List[Any]):
        self._col = {n: [] for n in data}
        self._colnames = data

    def row(self, data: List[Any]):
        for i, item in enumerate(data):
            key = self._colnames[i]
            if isinstance(item, list):
                item = ";".join(item)
            self._col[key].append(item)

    def close(self):
        df = pd.DataFrame(self._col)
        keys = list(df.keys())
        if self.CONTENT in keys:
            # split into content and not-content
            df_content = df[[self.NAME, self.CONTENT]]
            df = df[[k for k in keys if k != self.CONTENT]]
            # write each dataframe to separate sheet
            with pd.ExcelWriter(self._of) as ew:
                df.to_excel(ew, sheet_name="main")
                df_content.to_excel(ew, sheet_name="files")
        else:
            df.to_excel(self._of, index=False)


class JsonExporter(Exporter):
    """Dump to JSON.

    Note: Just use `json.dump(input-data)` since the input data is already in the right structure.
    """

    is_binary = False
    feed_rows = False

    def close(self):
        json.dump(self._data, self._of)


class GraphvizExporter(Exporter):

    feed_rows = False
    _open_file = False

    PREFIX_LEN = 3  # name prefix length, for node color/shape

    def _build(self):
        self._prefixes = set()
        self._keywords = {}
        self._keywords_r = {}
        self._filenames = {}
        self._connections = []

        for name, info in self._data.items():
            predicted = info[self.PREDICTED]
            file_abbr = f"F{len(self._filenames) + 1}"
            self._filenames[file_abbr] = name
            self._prefixes.add(name[: self.PREFIX_LEN])
            for keyword in predicted:
                kw_abbr = self._keywords_r.get(keyword, None)
                if kw_abbr is None:
                    kw_abbr = f"K{len(self._keywords) + 1}"
                    self._keywords[kw_abbr] = keyword
                    self._keywords_r[keyword] = kw_abbr
                self._connections.append((file_abbr, kw_abbr))

    def _pick_option(self, value, values, options):
        index = values.index(value)
        if index == -1:
            raise KeyError(value)
        return options[index % len(options)]

    def _graph(self):
        g = Graph(engine="neato")
        g.attr("graph", overlap="prism")

        prefixes = list(self._prefixes)

        node_colors, color_list = {}, ["#EE99AA", "#99AAEE"]
        for pfx in prefixes:
            node_colors[pfx] = self._pick_option(pfx, prefixes, color_list)

        # create nodes for files
        for abbr, name in self._filenames.items():
            prefix = name[: self.PREFIX_LEN]
            shape = "oval"
            color = node_colors[prefix]
            g.node(
                abbr,
                label=name,
                shape=shape,
                fillcolor=color,
                color=color,
                style="filled",
            )

        # create nodes for predicted keywords
        for abbr, name in self._keywords.items():
            g.node(abbr, label=name, shape="box")

        # connect nodes with edges
        for src, dst in self._connections:
            g.edge(src, dst)

        # create graph (or save for returning later)
        if self._path:
            self._graph = None
            sfx = self._path.suffix.lower()
            if sfx == ".svg":
                graph_format = "svg"
            elif sfx == ".png":
                graph_format = "png"
            elif sfx == ".pdf":
                graph_format = "pdf"
            else:
                raise ValueError("Output file must end in .svg, .png, or .pdf")
            filename = str(self._path.resolve())[:-4]  # strip suffix
            g.render(filename, format=graph_format, cleanup=True)
        else:
            self._graph = g  # return from getvalue()

    def close(self):
        self._build()
        self._graph()

    def getvalue(self):
        return "" if self._graph is None else self._graph


_exporters = {
    ExportFormat.EXCEL: ExcelExporter,
    ExportFormat.JSON: JsonExporter,
    ExportFormat.GRAPHVIZ: GraphvizExporter,
}


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
) -> Optional[str]:
    """Export input data to a file.

    Args:
        data: format = `{"item_name": {"value1_name": value1, ..}, ...}`
        output_filename: Output file name or path. If None, return as a string; if empty string or "-" write to standard output.
        output_format: Output format

    Raises:
        ValueError: Nothing to export
        ValueError: Unknown export format in `output_format`
        ValueError: Cannot write binary data to standard output
    """
    if not data:
        raise ValueError("Nothing to export")
    try:
        exporter_class = get_exporter_class(output_format)
    except KeyError:
        raise ValueError("Cannot get exporter class")
    exporter = exporter_class(output_filename, data=data)
    # export each row (if class wants it)
    if exporter.feed_rows:
        hdr = None
        for name, values in data.items():
            if hdr is None:
                hdr = list(values.keys())
                exporter.header([Exporter.NAME] + hdr)
            row = [name] + [values[k] for k in hdr]
            exporter.row(row)
    else:
        exporter.header(None)
    # close it
    exporter.close()
    if exporter.as_string:
        return exporter.getvalue()
