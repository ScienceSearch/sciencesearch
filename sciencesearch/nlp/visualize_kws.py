
import json
from pathlib import Path
from abc import ABC, abstractmethod
import webbrowser 
import os 
from nltk.tokenize import word_tokenize
from sciencesearch.nlp.models import CaseSensitiveStemmer
from sciencesearch.nlp.search import Searcher


class KWS_Visualizer(ABC):
    """ Abstract base class for keyword visualization """

    def __init__(self, txt_filepath: str = None, text: str = None):
        self.caseSS = CaseSensitiveStemmer()
        self.text = ""
        
        if txt_filepath:
            try:
                with open(txt_filepath, 'r') as file:
                    file_content = file.read()
                    self.text = file_content
            except FileNotFoundError:
                print(f"Error: File not found at '{txt_filepath}'")
            except Exception as e:
                    print(f"An error occurred: {e}")
        elif text:
            self.text = text
        self.tokens = word_tokenize(self.text)
        self.stemmed_tokens_text = [self.caseSS.stem(token) for token in self.tokens]
        self.max_kw_length = 0 
        self.html_content = ""


    def highlight_tokens_html(self):
        html_content = "<div class='highlighted-text'>"
        i = 0
        while i < len(self.tokens):
            matched = False
            for n in range(min(self.max_kw_length, len(self.tokens) - i), 0, -1):
                if i + n <= len(self.tokens):
                    ngram_text = [word.lower() for word in self.stemmed_tokens_text[i:i + n]]
                    ngram = tuple(ngram_text)
                    is_matched, styled_str = self.style_text(self.tokens[i:i + n], ngram)
                    if is_matched:
                        html_content += styled_str
                        i += n
                        matched = True
                        break
            if not matched:
                html_content += f"{self.tokens[i]} "
                i += 1
        html_content += "</div>"
        self.html_content = html_content
        return html_content
    
    @abstractmethod
    def style_text(self, tokens, ngram):
        pass



class SingleSet_Visualizer(KWS_Visualizer):

    def __init__(self, keywords: list[str], class_name: str = "keyword", text: str = None, txt_filepath: str = None):
        super().__init__(text=text, txt_filepath=txt_filepath)
        self.keywords = keywords
        self.class_name = class_name
        self.stemmed_kw_set = set()
        # self.stemmed_keywords = set([tuple(super.caseSS.stem(word_tokenize(phrase.lower()))) for phrase in keywords])
        for phrase in keywords:
            tokenized_phrase = word_tokenize(phrase.lower())
            stemmed_phrase = tuple(self.caseSS.stem(word) for word in tokenized_phrase)
            self.stemmed_kw_set.add(stemmed_phrase)
        
        self.max_kw_length = max([len(word_tokenize(phrase)) for phrase in keywords], default=1)


    def style_text(self, tokens, ngram):
        if ngram in self.stemmed_kw_set:
            styled_str = f"<span class='{self.class_name}'>{' '.join(tokens)}</span> "
            return True, styled_str
        else:
            return False, ""


class MultiSet_Visualizer(KWS_Visualizer):

    def __init__(self, keywords_dict: dict[str, list[str]], text: str = None, txt_filepath = None):
        super().__init__(text=text, txt_filepath=txt_filepath)
        self.keywords_dict = keywords_dict
        
        # Process each set of keywords
        self.stemmed_kw_sets = {}
        max_length = 1

        for set_name, keywords in keywords_dict.items():
            self.stemmed_kw_sets[set_name] = set()
            for phrase in keywords:
                phrase_tokens = word_tokenize(phrase.lower())
                stemmed_phrase = tuple(self.caseSS.stem(token) for token in phrase_tokens)
                self.stemmed_kw_sets[set_name].add(stemmed_phrase)
                max_length = max(max_length, len(phrase_tokens))
        
        self.max_kw_length = max_length
    

    def style_text(self, tokens, ngram):
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


class HTMLBuilder:

    def __init__(self, visualizer: KWS_Visualizer, filename: str, title: str, 
                 css_styles: dict[str, str] = None):
        self.visualizer = visualizer
        self.filename = filename
        self.title = title
        self.html = ""
        self.css_styles = css_styles or {}

    def get_highlighted_html(self):
        html_body =  self.visualizer.highlight_tokens_html()
        html_output = self.generate_html(self.title, html_body)
        self.html = html_output


    def write_file_and_run(self):
        file = open(self.filename,'w')
        file.write(self.html) 
        file.close() 
        fp = 'file:///'+os.getcwd()+'/' + self.filename
        webbrowser.open_new_tab(fp) 

    def generate_html(self, title: str = "" ,body_content: str = ""):

        legend = ""
        if isinstance(self.visualizer, MultiSet_Visualizer):
            legend = """
            <div class="legend">
                <h2>Legend</h2>
                <p><span class="kw-training">Only in training keywords set</span></p>
                <p><span class="kw-default">Only in default keywords set</span></p>
                <p><span class="kw-tuned">Only in tuned keyword set</span></p>
                <p><span class="kw-training kw-default">In both training and default keyword sets</span></p>
                <p><span class="kw-training kw-tuned">In both training and tuned keyword sets</span></p>
                <p><span class="kw-default kw-tuned">In both default and tuned keyword sets</span></p>
                <p><span class="kw-training kw-default kw-tuned">In all three keyword sets</span></p>
            </div>
            """

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
             <h1>Experiment: {self.title}</h1>
            {legend}
            {body_content}

        </body>
        </html>
        """
        return html
    
class JsonView:
    def __init__(self, searcher: Searcher):
        self.searcher = searcher
        self._predicted_keywords = searcher.predicted_keywords.copy()
        self._file_keywords = searcher.file_keywords.copy()
        self._training_keywords = searcher.training_keywords.copy()


    def save_predicted_keywords(self, filename: str):
        res = self._predicted_keywords.copy()

        with open(filename, 'w') as file:
            json.dump(res, file, indent=4)

    def save_file_keywords(self, filename: str):
        res = self._training_keywords.copy()


        with open(filename, 'w') as file:
            json.dump(res, file, indent=4)


    def save_training_keywords(self, filename: str):
        res = self._training_keywords.copy()

        with open(filename, 'w') as file:
            json.dump(res, file, indent=4)


    def save_all_keyword_sets(self,  filename: str):
        
        res = {}
        for fn, keywords in self._file_keywords.items():
            all_kws = {'training':self._training_keywords.get(fn, []),
                        'tuned': self._predicted_keywords.get(fn, [])}
            res[fn] = all_kws    
        with open(filename, 'w') as file:
            file.write(json.dumps(res))

    def visualize_from_config(config_file, is_singleset: bool, json_file: json, save_file_prefix: str):
        conf = json.load(open(config_file))
        training = conf["training"]
        file_dir = Path(training["directory"])

        with open(json_file) as json_data:
            data = json.load(json_data)        
        if is_singleset:
            for textfilename, keywords in data.items():
                filepath = f"{file_dir}/{textfilename}"
                file = textfilename[:textfilename.find('.')]
                sskw = SingleSet_Visualizer(keywords=keywords, txt_filepath = filepath)
                htmlbuilder = HTMLBuilder(visualizer=sskw, filename=f"{save_file_prefix}_{file}", title=textfilename)
                htmlbuilder.get_highlighted_html()
                htmlbuilder.write_file_and_run()
        else:
            for textfilename, keywords in data.items():
                filepath = f"{file_dir}/{textfilename}"
                file = textfilename[:textfilename.find('.')]
                mskw = MultiSet_Visualizer(keywords_dict=keywords, txt_filepath = filepath)
                htmlbuilder = HTMLBuilder(visualizer=mskw, filename=f"{save_file_prefix}_{file}", title=textfilename)
                htmlbuilder.get_highlighted_html()
                htmlbuilder.write_file_and_run()
