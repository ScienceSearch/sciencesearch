
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sciencesearch.nlp.models import CaseSensitiveStemmer
from abc import ABC, abstractmethod

import webbrowser 
import os 


class KWS_Visualizer(ABC):
    """ Abstract """

    def __init__(self, text: str, filename: str):
        self.caseSS = CaseSensitiveStemmer()
        self.text = text
        self.tokens = word_tokenize(text)
        self.stemmed_token_text = [self.caseSS.stem(token) for token in self.tokens]
        self.filename = filename

    def highlight_tokens_html(self):
        html_content = "<p>"
        i = 0
        while i < len(self.tokens):
            matched = False
            for n in range(self.max_kw_length, 0, -1):
                if i + n <= len(self.tokens):
                    ngram_text = [word.lower() for word in self.stemmed_tokens_text[i:i + n]]
                    ngram = tuple(ngram_text)
                    is_matched, styled_str = self.style_text(self,self.tokens[i:i + n], ngram)
                    if is_matched:
                        html_content += styled_str
                        i += n
                        break
            if not matched:
                html_content += f"{self.tokens[i]} "
                i += 1
        html_content += "</p>"
    
    @abstractmethod
    def style_text(self, tokens: str, ngam):
        pass


class SingleSet_Visualizer(KWS_Visualizer):

    def __init__(self, keywords: list[str], **kwargs):
        super().__init__(**kwargs)
        self.keywords = keywords
        self.stemmed_keywords = set([tuple(super.caseSS.stem(word_tokenize(phrase.lower()))) for phrase in keywords])
        self.max_kw_length = max(len(phrase) for phrase in self.stemmed_keywords)

    def style_text(self, tokens: str, ngram ):
        if ngram in self.stemmed_kw_set:
            styled_str += f"<strong>{' '.join(tokens)}</strong> "
            return True, styled_str
        else:
            return False, ""



class MultiSet_Visualizar(KWS_Visualizer):

        def __init__(self, keywords: dict[str, list], **kwargs):
            super().__init__(**kwargs)
            self.keywords = keywords
        
        def style_text(self, tokens, ngram: tuple):
            css = ""
            if ngram in self.stemmed_kw_set_dict.get('manual'):
                css = "color:#B99512;"
            if ngram in self.stemmed_kw_set_dict.get('tuned'):
                css ="color:#2F539B;"
            if ngram in self.stemmed_kw_set_dict.get('default'):
                css = "color:#B83C08;"
            if ngram in self.stemmed_kw_set_dict.get('manual') and ngram in self.stemmed_kw_set_dict.get('default') and ngram not in self.stemmed_kw_set_dict.get('tuned'):
                css = 'color:#FF7722;'
            if ngram in self.stemmed_kw_set_dict.get('tuned') and ngram in self.stemmed_kw_set_dict.get('default') and ngram not in self.stemmed_kw_set_dict.get('manual'):
                css = 'color:#7D0552;'
            if ngram in self.stemmed_kw_set_dict.get('manual') and ngram in self.stemmed_kw_set_dict.get('tuned') and ngram not in self.stemmed_kw_set_dict.get('default'):
                css = 'color:#1AA260;'
            if ngram in self.stemmed_kw_set_dict.get('manual') and ngram in self.stemmed_kw_set_dict.get('tuned') and ngram in self.stemmed_kw_set_dict.get('default'):
                css = 'color:#000000;'
            
            if style_str:
                style_str+="font-weight:bold"
                styled_str += f"<span style = '{css}'>{' '.join(tokens[i:i + n])}</span> "
                return True, styled_str
            else:
                return False, ""
            


class HTMLBuilder:

    def __init__(self, visualizer: KWS_Visualizer, filename: str, title: str, is_multiple_sets: bool):
        self.visualizer = visualizer
        self.filename = filename
        self.title = title
        self.is_multiple_sets = is_multiple_sets
        self.html = ""


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

    def generate_html(self, body_content: str):

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
        </head>
        <body>
            <h1> experiment {self.title} </h1>
            <p>  
                <span style ='color:#B99512;font-weight:bold'> only in manual keywords set </span><br />
                <span style ='color:#B83C08;font-weight:bold'> only in default keywords set </span><br />
                <span style ='color:#2F539B;font-weight:bold'> only in tuned keyword set </span><br />
                <span style ='color:#FF7722;font-weight:bold'> in both manual and default keyword set </span><br />
                <span style ='color:#006A4E;font-weight:bold'> in both manual and tuned keyword set </span><br />
                <span style ='color:#7D0552;font-weight:bold'> in both default and tuned keyword set </span><br />
                <span style ='color:#000000;font-weight:bold'> in all manual labeled, default, and tuned keyword set</span><br />
            </p>

            {body_content}

        </body>
        </html>
        """
        return html