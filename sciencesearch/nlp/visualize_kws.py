import json
from pathlib import Path
from abc import ABC, abstractmethod
import webbrowser
import os
from nltk.tokenize import word_tokenize
from sciencesearch.nlp.models import CaseSensitiveStemmer
from sciencesearch.nlp.search import Searcher

"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""


class KeywordsVisualizer(ABC):
    """Abstract base class for keyword visualization.

    A KeywordVisualizer creates HTML text with snippets highlighted based on defined keywords.
    This class handles the common functionality of tokenizing text, stemming words, and
    generating HTML output with highlighted keywords.

    Subclasses should:
        - Call super().__init__() in their constructor with appropriate parameters
        - Override the method style_text() to define their own styling of keywords
        - Override the method get_legend() to return appropriate legend HTML

    Attributes:
        text (str): The input text to be highlighted.
        title (str): Title for the visualization.
        html_content (str): Generated HTML content with highlighted keywords (read-only).
    """

    def __init__(self, txt_filepath: str = None, text: str = None, title: str = None):
        """Initializes the KeywordsVisualizer.

        Args:
            txt_filepath (str, optional): Path to text file containing input text.
                If provided, text will be read from this file.
            text (str, optional): Input text string to be highlighted.
                Used if txt_filepath is not provided.
            title (str, optional): Title header for the visualization output.

        Note:
            Either txt_filepath or text should be provided, but not both.
            If txt_filepath is provided, it takes precedence over text parameter.
        """
        # public
        self.text = text
        self.title = title

        # private
        self._case_sensitive_stemmer = CaseSensitiveStemmer()
        self._tokens = []
        self._stemmed_tokens_text = []
        self._html_content = None

        self._load_text_from_file(txt_filepath)
        self._tokenize_text()
        self._get_formatted_html_body()

    @property
    def html_content(self):
        """str: HTML block styled with appropriately highlighted keywords (read-only)."""
        return self._html_content

    def _load_text_from_file(self, txt_filepath: str):
        if txt_filepath:
            try:
                with open(txt_filepath, "r", encoding="utf-8") as file:
                    self.text = file.read()
            except FileNotFoundError:
                print(f"Error: File not found at '{txt_filepath}'")
            except Exception as e:
                print(f"An error occurred: {e}")

    def _tokenize_text(self):
        if self.text:
            self._tokens = word_tokenize(self.text)
            self._stemmed_tokens_text = [
                self._case_sensitive_stemmer.stem(token) for token in self._tokens
            ]

    def _get_formatted_html_body(self, body_content: str = ""):
        legend = self.get_legend()
        body_content = self._highlight_tokens_html()

        self._html_content = f"""<h1>Experiment: {self.title}</h1>
            {legend}
            {body_content}"""

    def _highlight_tokens_html(self):
        html_content = "<div class='highlighted-text'>"
        i = 0
        while i < len(self._tokens):
            matched = False
            for n in range(min(self._max_kw_length, len(self._tokens) - i), 0, -1):
                if i + n <= len(self._tokens):
                    ngram_text = [
                        word.lower() for word in self._stemmed_tokens_text[i : i + n]
                    ]
                    ngram = tuple(ngram_text)
                    is_matched, styled_str = self.style_text(
                        self._tokens[i : i + n], ngram
                    )
                    if is_matched:
                        html_content += styled_str
                        i += n
                        matched = True
                        break
            if not matched:
                html_content += f"{self._tokens[i]} "
                i += 1
        html_content += "</div>"
        return html_content

    @abstractmethod
    def style_text(self, tokens, ngram):
        """Adds CSS styling to keywords in the text.

        Args:
            tokens (list[str]): Original tokens that matched a keyword.
            ngram (tuple[str]): Stemmed and lowercased version of the tokens.

        Returns:
            tuple[bool, str]: A tuple containing:
                - bool: True if the ngram matches a keyword, False otherwise.
                - str: HTML string with styled text if matched, empty string otherwise.
        """
        pass

    @abstractmethod
    def get_legend(self):
        """Creates a legend explaining the meaning of highlighted text.

        Returns:
            str: HTML string containing the legend for the visualizer.
        """
        pass


class SingleSetVisualizer(KeywordsVisualizer):
    """Subclass of KeywordVisualizer for highlighting a single set of keywords.

    Attributes:
        keywords (list[str]): List of keyword phrases to highlight.
        style_class_name (str): CSS class name to apply to highlighted keywords.
    """

    def __init__(
        self,
        keywords: list[str],
        style_class_name: str = "keyword",
        txt_filepath: str = None,
        text: str = None,
        title: str = None,
    ):
        """Initializes SingleSetVisualizer.

        Args:
            keywords (list[str]): List of keyword phrases to be highlighted in the text.
            style_class_name (str, optional): CSS class name for styling highlighted keywords.
                Defaults to "keyword".
            txt_filepath (str, optional): Path to text file containing input text.
            text (str, optional): Input text string to be highlighted.
            title (str, optional): Title for the visualization output.
        """
        self.keywords = keywords.copy()
        self.class_name = style_class_name
        self._stemmed_kw_set = set()

        self._process_keywords()

        super().__init__(text=text, txt_filepath=txt_filepath, title=title)

    def _process_keywords(self):
        temp_stemmer = CaseSensitiveStemmer()

        for phrase in self.keywords:
            tokenized_phrase = word_tokenize(phrase.lower())
            stemmed_phrase = tuple(temp_stemmer.stem(word) for word in tokenized_phrase)
            self._stemmed_kw_set.add(stemmed_phrase)

        self._max_kw_length = max(
            [len(word_tokenize(phrase)) for phrase in self.keywords], default=1
        )

    def style_text(self, tokens, ngram):
        """See base class."""
        if ngram in self._stemmed_kw_set:
            styled_str = f"<span class='{self.class_name}'>{' '.join(tokens)}</span> "
            return True, styled_str
        else:
            return False, ""

    def get_legend(self):
        """See base class."""
        return """
            <div class="legend">
                <h2>Legend</h2>
                <p><span class="keyword">Keywords in document</span></p>
            </div>
            """


class MultiSetVisualizer(KeywordsVisualizer):
    """Subclass of KeywordVisualizer for highlighting multiple sets of keywords with different styles.

    Attributes:
        keywords_dict (dict[str, list[str]]): Dictionary mapping set names to keyword lists.
    """

    def __init__(
        self,
        keywords_dict: dict[str, list[str]],
        text: str = None,
        txt_filepath=None,
        title: str = None,
    ):
        """Initializes the MultiSetVisualizer.

        Args:
            keywords_dict (dict[str, list[str]]): Dictionary where keys are set names
                and values are lists of keyword phrases for that set.
            txt_filepath (str, optional): Path to text file containing input text.
            text (str, optional): Input text string to be highlighted.
            title (str, optional): Title for the visualization output.
        """
        self.keywords_dict = {k: v.copy() for k, v in keywords_dict.items()}
        # Process each set of keywords
        self._stemmed_kw_sets = {}
        self._process_keyword_sets()

        super().__init__(text=text, txt_filepath=txt_filepath, title=title)

    def _process_keyword_sets(self):
        temp_stemmer = CaseSensitiveStemmer()
        max_length = 1

        for set_name, keywords in self.keywords_dict.items():
            self._stemmed_kw_sets[set_name] = set()
            for phrase in keywords:
                phrase_tokens = word_tokenize(phrase.lower())
                stemmed_phrase = tuple(
                    temp_stemmer.stem(token) for token in phrase_tokens
                )
                self._stemmed_kw_sets[set_name].add(stemmed_phrase)
                max_length = max(max_length, len(phrase_tokens))

        self._max_kw_length = max_length

    def style_text(self, tokens, ngram):
        """See base class."""
        # Determine which sets this n-gram belongs to
        matching_sets = []
        for set_name, stemmed_set in self._stemmed_kw_sets.items():
            if ngram in stemmed_set:
                matching_sets.append(set_name)

        if matching_sets:
            # Create a class name based on which sets matched
            class_name = " ".join([f"kw-{set_name}" for set_name in matching_sets])
            styled_str = f'<span class="{class_name}">{" ".join(tokens)}</span> '
            return True, styled_str
        return False, ""

    def get_legend(self):
        """See base class."""
        return """
            <div class="legend">
                <h2>Legend</h2>
                <p><span class="kw-training">Only in training keywords set</span></p>
                <p><span class="kw-tuned">Only in tuned keyword set</span></p>
                <p><span class="kw-training kw-tuned">In both training and tuned keyword sets</span></p>
            </div>
            """


class HTMLBuilder:
    """Builds complete HTML documents with keyword visualizations.

    This class takes one or more KeywordsVisualizer objects and generates a complete
    HTML document with proper structure, CSS styling, and the ability to open the
    result in a web browser.

    Attributes:
        visualizers (list[KeywordsVisualizer]): List of visualizer objects to include.
        filename (str): Output filename for the HTML document.
        title (str): Title for the HTML document.
        css_styles_filepath (str): Path to CSS stylesheet file.
    """

    def __init__(
        self,
        visualizers: list[KeywordsVisualizer],
        filename: str,
        title: str,
        css_styles_filepath: str = "shared/keyword_vis.css",
    ):
        """Initializes HTMLBuilder. HTML generation occurs automatically during initialization.

        Args:
            visualizers (list[KeywordsVisualizer]): List of KeywordsVisualizer objects
                whose content will be included in the HTML document.
            filename (str): Name of the output HTML file.
            title (str): Title for the HTML document (appears in browser tab and page).
            css_styles_filepath (str, optional): Relative path to CSS stylesheet.
                Defaults to 'shared/keyword_vis.css'.
        """
        self.visualizers = visualizers.copy()
        self.filename = filename
        self.title = title
        self.css_styles_filepath = css_styles_filepath

        self.html = ""
        self._generate_html()

    def _get_highlighted_html(self):
        html_body = ""
        for visualizer in self.visualizers:
            html_body += visualizer.html_content
        return html_body

    def write_file_and_run(self):
        """Writes HTML to file and opens it in the default web browser.

        The method creates the HTML file and automatically opens it in a new browser tab
        using the system's default web browser.
        """
        with open(self.filename, "w") as file:
            file.write(self.html)
        fp = "file:///" + os.getcwd() + "/" + self.filename
        webbrowser.open_new_tab(fp)

    def _generate_html(self):
        body_content = self._get_highlighted_html()
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link rel="stylesheet" type= "text/css" href="../shared/keyword_vis.css">
            
            <title>{self.title}</title>
        </head>
        <body>
        {body_content}
    
        </body>
        </html>
        """
        self.html = html


class JsonView:
    """Utility class for saving and visualizing keyword data from Searcher objects.

    This class provides methods to extract keyword data from Searcher objects,
    save it in JSON format, and create visualizations from saved JSON data.
    It handles three types of keyword sets: predicted, file-based, and training keywords.

    Attributes:
        searcher (Searcher): The searcher object containing keyword data.
    """

    def __init__(self, searcher: Searcher):
        """Initializes JsonView with a Searcher object.

        Args:
            searcher (Searcher): Searcher object containing keyword data to be exported.
        """
        self.searcher = searcher
        self.__predicted_keywords = searcher.predicted_keywords.copy()
        self.__file_keywords = searcher.file_keywords.copy()
        self.__training_keywords = searcher.training_keywords.copy()

    @property
    def predicted_keywords(self):
        """Saves predicted keywords to a JSON file.

        Args:
            filename (str): Path to the output JSON file.
        """
        JsonView._print_keywords(self.__predicted_keywords)
        return self.__predicted_keywords.copy()

    @property
    def file_keywords(self):
        """Saves file keywords to a JSON file.

        Args:
            filename (str): Path to the output JSON file.
        """
        JsonView._print_keywords(self.__file_keywords)
        return self.__file_keywords.copy()

    @property
    def training_keywords(self):
        """Prints training keywords
        dict: A dictionary of input file names and their training keywords
        """

        JsonView._print_keywords(self.__training_keywords)
        return self.__training_keywords.copy()

    @staticmethod
    def _print_keywords(keywords: dict):
        for f, k in keywords.items():
            print(f"{f} => {', '.join(k)}")

    def save_predicted_keywords(self, filename: str):
        """Saves predicted keywords to a JSON file.

        Args:
            filename (str): Path to the output JSON file.
        """
        res = self.__predicted_keywords.copy()

        with open(filename, "w") as file:
            json.dump(res, file, indent=4)

    def save_file_keywords(self, filename: str):
        """Saves file keywords to a JSON file.

        Args:
            filename (str): Path to the output JSON file.
        """
        res = self.__training_keywords.copy()

        with open(filename, "w") as file:
            json.dump(res, file, indent=4)

    def save_training_keywords(self, filename: str):
        """Saves training keywords to a JSON file.

        Args:
            filename (str): Path to the output JSON file.
        """
        res = self.__file_keywords.copy()

        with open(filename, "w") as file:
            json.dump(res, file, indent=4)

    def save_all_keyword_sets(self, filename: str):
        """Saves all keyword sets in a combined JSON structure.

        Creates a JSON file where each filename maps to a dictionary containing
        both training and tuned (predicted) keyword sets.

        Args:
            filename (str): Path to the output JSON file.

        Note:
            The output structure is:
            {
                "filename1.txt": {
                    "training": ["keyword1", "keyword2", ...],
                    "tuned": ["keyword3", "keyword4", ...]
                },
                ...
            }
        """
        res = {}
        for fn, keywords in self.__file_keywords.items():
            all_kws = {
                "training": self.__training_keywords.get(fn, []),
                "tuned": self.__predicted_keywords.get(fn, []),
            }
            res[fn] = all_kws
        with open(filename, "w") as file:
            file.write(json.dumps(res))

    @staticmethod
    def visualize_from_config(config_file, json_file: str, save_filename: str):
        """Creates HTML visualizations from configuration and JSON keyword files.

        This static method reads a configuration file to determine text file locations
        and a JSON file containing keyword data, then generates HTML visualizations
        for each text file with its associated keywords.

        Args:
            config_file (str): Path to JSON configuration file containing training directory.
                Expected structure: {"training": {"directory": "path/to/texts"}}
            json_file (str): Path to JSON file containing keyword data.
                Can contain either single keyword lists or multi-set keyword dictionaries.
            save_file_prefix (str): Prefix for generated HTML filenames.

        Generated HTML files are automatically opened in the default web browser.
        """
        conf = json.load(open(config_file))
        training = conf["training"]
        file_dir = Path(training["directory"])

        with open(json_file) as json_data:
            data = json.load(json_data)
        visualizers = []

        for textfilename, keywords in data.items():
            filepath = f"{file_dir}/{textfilename}"
            file = textfilename[: textfilename.find(".")]
            visualizer = None
            if isinstance(keywords, list):
                visualizer = SingleSetVisualizer(
                    keywords=keywords, txt_filepath=filepath, title=file
                )
            elif isinstance(keywords, dict):
                visualizer = MultiSetVisualizer(
                    keywords_dict=keywords, txt_filepath=filepath, title=file
                )
            visualizers.append(visualizer)
        htmlbuilder = HTMLBuilder(
            visualizers=visualizers,
            filename=f"{save_filename}",
            title="Highlighted Keywords",
        )
        htmlbuilder.write_file_and_run()
