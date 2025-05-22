
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from sciencesearch.nlp.models import CaseSensitiveStemmer
from abc import ABC, abstractmethod
from sciencesearch.nlp.search import Searcher
import json

import webbrowser 
import os 


class KWS_Visualizer(ABC):
    """ Abstract base class for keyword visualization """

    def __init__(self, txt_filepath: str = None, text: str = None):
        self.caseSS = CaseSensitiveStemmer()
        self.text = ""
        
        if txt_filepath:
            try:
                with open(txt_filepath, 'r') as file:
                    file_content = file.read()
                    print(type(file_content))
                    print(file_content)
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
    
        
    # def style_text(self, tokens, ngram: tuple):
    #     css = ""
    #     if ngram in self.stemmed_kw_set_dict.get('manual'):
    #         css = "color:#B99512;"
    #     if ngram in self.stemmed_kw_set_dict.get('tuned'):
    #         css ="color:#2F539B;"
    #     if ngram in self.stemmed_kw_set_dict.get('default'):
    #         css = "color:#B83C08;"
    #     if ngram in self.stemmed_kw_set_dict.get('manual') and ngram in self.stemmed_kw_set_dict.get('default') and ngram not in self.stemmed_kw_set_dict.get('tuned'):
    #         css = 'color:#FF7722;'
    #     if ngram in self.stemmed_kw_set_dict.get('tuned') and ngram in self.stemmed_kw_set_dict.get('default') and ngram not in self.stemmed_kw_set_dict.get('manual'):
    #         css = 'color:#7D0552;'
    #     if ngram in self.stemmed_kw_set_dict.get('manual') and ngram in self.stemmed_kw_set_dict.get('tuned') and ngram not in self.stemmed_kw_set_dict.get('default'):
    #         css = 'color:#1AA260;'
    #     if ngram in self.stemmed_kw_set_dict.get('manual') and ngram in self.stemmed_kw_set_dict.get('tuned') and ngram in self.stemmed_kw_set_dict.get('default'):
    #         css = 'color:#000000;'
        
    #     if style_str:
    #         style_str+="font-weight:bold"
    #         styled_str += f"<span style = '{css}'>{' '.join(tokens[i:i + n])}</span> "
    #         return True, styled_str
    #     else:
    #         return False, ""
        


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
        # css = """
        # .highlight { background-color: yellow; font-weight: bold; }
        
        # /* For multi-set visualizer */
        # .kw-manual { color: #B99512; font-weight: bold; }
        # .kw-default { color: #B83C08; font-weight: bold; }
        # .kw-tuned { color: #2F539B; font-weight: bold; }
        
        # /* Combined classes */
        # .kw-manual.kw-default:not(.kw-tuned) { color: #FF7722; font-weight: bold; }
        # .kw-manual.kw-tuned:not(.kw-default) { color: #006A4E; font-weight: bold; }
        # .kw-default.kw-tuned:not(.kw-manual) { color: #7D0552; font-weight: bold; }
        # .kw-manual.kw-default.kw-tuned { color: #000000; font-weight: bold; }
        # """


        legend = ""
        if isinstance(self.visualizer, MultiSet_Visualizer):
            legend = """
            <div class="legend">
                <h2>Legend</h2>
                <p><span class="kw-manual">Only in manual keywords set</span></p>
                <p><span class="kw-default">Only in default keywords set</span></p>
                <p><span class="kw-tuned">Only in tuned keyword set</span></p>
                <p><span class="kw-manual kw-default">In both manual and default keyword sets</span></p>
                <p><span class="kw-manual kw-tuned">In both manual and tuned keyword sets</span></p>
                <p><span class="kw-default kw-tuned">In both default and tuned keyword sets</span></p>
                <p><span class="kw-manual kw-default kw-tuned">In all three keyword sets</span></p>
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
    

class ResultsJson:
    def __init__(self):
        pass
    def save_results(self,searcher: Searcher, filename: str):
        keyword_mapping = searcher.file_keywords
        print("km", keyword_mapping)
        with open(filename, 'w') as file:
            json.dump(keyword_mapping, file, indent=4)


            

