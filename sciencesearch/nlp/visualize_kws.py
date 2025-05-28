import json
from pathlib import Path
from abc import ABC, abstractmethod
from typing_extensions import override
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

    A KeywordVisualizer creates text, with snippets highlighted based on defined keywords. 

    Sublasses should:
      - define their own styling of keywords
      - call `super().__init__(text_filepath: str for filepath to text file, text: str)` in their constructor
        where depending on the source, one should be None
      - override the method `style_text()` to return styled keyword html
      - override the method `get_legend()` to return the legend for the type of visualizer

      Attributes:
        html_content: styled html block of highlighted keywords
    """

    def __init__(self, txt_filepath: str = None, text: str = None, title: str = None):
        """Base constructer.

        Args:
            text_filepath: filepath to text file path with original text input to be highlighted
            text: string of text to be highlighted
        """
        self.caseSS = CaseSensitiveStemmer()
        self.text = text
        self.title = title

        if txt_filepath:
            try:
                with open(txt_filepath, "r") as file:
                    file_content = file.read()
                    self.text = file_content
            except FileNotFoundError:
                print(f"Error: File not found at '{txt_filepath}'")
            except Exception as e:
                print(f"An error occurred: {e}")
        if self.text:  # Only tokenize if we have text
            self.tokens = word_tokenize(self.text)
            self.stemmed_tokens_text = [self.caseSS.stem(token) for token in self.tokens]
        else:
            self.tokens = []
            self.stemmed_tokens_text = []
            
        self.get_formatted_html_body()

    @property
    def html_content(self):
        """HTML block styled with appropriately highlighted keywords"""
        return self._html_content

    def _highlight_tokens_html(self):
        html_content = "<div class='highlighted-text'>"
        i = 0
        while i < len(self.tokens):
            matched = False
            for n in range(min(self.max_kw_length, len(self.tokens) - i), 0, -1):
                if i + n <= len(self.tokens):
                    ngram_text = [
                        word.lower()
                        for word in self.stemmed_tokens_text[i : i + n]
                    ]
                    ngram = tuple(ngram_text)
                    is_matched, styled_str = self.style_text(
                        self.tokens[i : i + n], ngram
                    )
                    if is_matched:
                        html_content += styled_str
                        i += n
                        matched = True
                        break
            if not matched:
                html_content += f"{self.tokens[i]} "
                i += 1
        html_content += "</div>"
        return html_content

    @abstractmethod
    def style_text(self, tokens, ngram):
        """Adds CSS styling to keywords in the text"""
        pass

    @abstractmethod
    def get_legend(self):
        """Creates a legend for the meaning of highighted text for the visualizer"""
        pass

    def get_formatted_html_body(self, body_content: str = ""):
        legend = self.get_legend()
        body_content = self._highlight_tokens_html()

        self._html_content =  f"""<h1>Experiment: {self.title}</h1>
            {legend}
            {body_content}"""


class SingleSetVisualizer(KeywordsVisualizer):
    """Subclass of KeywordVisualizer that handles a single set of keywords for a text passage."""

    def __init__(
        self,
        keywords: list[str],
        style_class_name: str = "keyword",
        txt_filepath: str = None,
        text: str = None,
        title: str = None,
    ):
        """ Base constructer.

        Args:
            keywords: a list of strings to be highlighted in the passage
            style_class_name: the class within the css styling that should be applied to highlighted keywords
            text_filepath: filepath to text file path with original text input to be highlighted
            text: string of text to be highlighted
        """
        self.keywords = keywords
        self.class_name = style_class_name
        self.stemmed_kw_set = set()
        
        # Initialize the stemmer here since we need it before calling super().__init__
        temp_stemmer = CaseSensitiveStemmer()
        
        for phrase in keywords:
            tokenized_phrase = word_tokenize(phrase.lower())
            stemmed_phrase = tuple(temp_stemmer.stem(word) for word in tokenized_phrase)
            self.stemmed_kw_set.add(stemmed_phrase)

        self.max_kw_length = max(
                [len(word_tokenize(phrase)) for phrase in keywords], default=1
            )  
        super().__init__(text=text, txt_filepath=txt_filepath, title=title)
       
  
    @override
    def style_text(self, tokens, ngram):
        """See base class."""
        if ngram in self.stemmed_kw_set:
            styled_str = f"<span class='{self.class_name}'>{' '.join(tokens)}</span> "
            return True, styled_str
        else:
            return False, ""

    @override
    def get_legend(self):
        """See base class."""
        return """
            <div class="legend">
                <h2>Legend</h2>
                <p><span class="keyword">Keywords in document</span></p>
            </div>
            """


class MultiSetVisualizer(KeywordsVisualizer):
    """Subclass of KeywordVisualizer that handles a multiple set of keywords for a text passage."""

    def __init__(
        self, keywords_dict: dict[str, list[str]], text: str = None, txt_filepath=None, title: str = None,

    ):
        
        """ Base constructer.

        Args:
            keywords: a dictionary with a type of keyword key associated with list of strings to be highlighted in the passage
            text_filepath: filepath to text file path with original text input to be highlighted
            text: string of text to be highlighted
        """

        self.keywords_dict = keywords_dict

        # Process each set of keywords
        self.stemmed_kw_sets = {}
        max_length = 1
        
        # Initialize the stemmer here since we need it before calling super().__init__
        temp_stemmer = CaseSensitiveStemmer()
        for set_name, keywords in keywords_dict.items():
            self.stemmed_kw_sets[set_name] = set()
            for phrase in keywords:
                phrase_tokens = word_tokenize(phrase.lower())
                stemmed_phrase = tuple(
                    temp_stemmer.stem(token) for token in phrase_tokens
                )
                self.stemmed_kw_sets[set_name].add(stemmed_phrase)
                max_length = max(max_length, len(phrase_tokens))
        

        self.max_kw_length = max_length
        super().__init__(text=text, txt_filepath=txt_filepath, title=title)


    def style_text(self, tokens, ngram):
        """See base class."""
        # Determine which sets this n-gram belongs to
        matching_sets = []
        for set_name, stemmed_set in self.stemmed_kw_sets.items():
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
    """Can build and execute html"""

    def __init__(
        self,
        visualizers: list[KeywordsVisualizer],
        filename: str,
        title: str,
        css_styles_filepath: str = 'shared/keyword_vis.css'
    ):
        """"Base constructer. 

        Args:
            visualizer: A KeywordsVisualizer object to compile and run HTML off
            filename: the filename to the HTML to
            title: the title of the HTML
            css_styles_filepath: optional filepath as string to style text. Must follow default format.
        """
        self.visualizers = visualizers
        self.filename = filename
        self.title = title
        self.css_styles_filepath = css_styles_filepath
        self.__generate_html()

        
    def __get_highlighted_html(self):
        html_body = ""
        for visualizer in self.visualizers:
            html_body += visualizer.html_content
        return html_body

    def write_file_and_run(self):
        with open(self.filename, "w") as file:
            file.write(self.html)
        fp = "file:///" + os.getcwd() + "/" + self.filename
        webbrowser.open_new_tab(fp)

    def __generate_html(self):
        body_content = self.__get_highlighted_html()
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
    """"""
    def __init__(self, searcher: Searcher):
        """
        """
        self.searcher = searcher
        self.__predicted_keywords = searcher.predicted_keywords.copy()
        self.__file_keywords = searcher.file_keywords.copy()
        self.__training_keywords = searcher.training_keywords.copy()

    def save_predicted_keywords(self, filename: str):
        """ """
        res = self.__predicted_keywords.copy()

        with open(filename, "w") as file:
            json.dump(res, file, indent=4)

    def save_file_keywords(self, filename: str):
        """ """
        res = self.__training_keywords.copy()

        with open(filename, "w") as file:
            json.dump(res, file, indent=4)

    def save_training_keywords(self, filename: str):
        """ """
        res = self.__file_keywords.copy()

        with open(filename, "w") as file:
            json.dump(res, file, indent=4)

    def save_all_keyword_sets(self, filename: str):
        """ """

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
    def visualize_from_config(
        config_file, json_file: json, save_filename: str
    ):
        """
        
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
                    visualizer = SingleSetVisualizer(keywords=keywords, txt_filepath=filepath, title = file)
            if isinstance(keywords, dict):
                    visualizer = MultiSetVisualizer(keywords_dict=keywords, txt_filepath=filepath, title = file)
            visualizers.append(visualizer)
        htmlbuilder = HTMLBuilder(visualizers=visualizers, filename=f"{save_filename}",
                title='Highlighted Keywords')
        htmlbuilder.write_file_and_run()


        # if is_singleset:
        #     for textfilename, keywords in data.items():
        #         filepath = f"{file_dir}/{textfilename}"
        #         file = textfilename[: textfilename.find(".")]
        #         sskw = SingleSetVisualizer(keywords=keywords, txt_filepath=filepath)
        #         htmlbuilder = HTMLBuilder(
        #             visualizer=sskw,
        #             filename=f"{save_file_prefix}_{file}",
        #             title=textfilename,
        #         )
        #         htmlbuilder.get_highlighted_html()
        #         htmlbuilder.write_file_and_run()
        # else:
        #     for textfilename, keywords in data.items():
        #         filepath = f"{file_dir}/{textfilename}"
        #         file = textfilename[: textfilename.find(".")]
        #         mskw = MultiSetVisualizer(keywords_dict=keywords, txt_filepath=filepath)
        #         htmlbuilder = HTMLBuilder(
        #             visualizer=mskw,
        #             filename=f"{save_file_prefix}_{file}",
        #             title=textfilename,
        #         )
        #         htmlbuilder.get_highlighted_html()
        #         htmlbuilder.write_file_and_run()
