"""
Preprocess text.
"""

import logging
from pathlib import Path
from string import punctuation
from collections import Counter
import regex as re
import pandas as pd
from bs4 import BeautifulSoup

_log = logging.getLogger(__name__)


class Preprocessor:
    """Preprocess text to remove extraneous symbols"""

    def __init__(self):
        """Constructor."""
        # chain together cleaning functions in this order
        self._functions = [
            self._remove_newlines,
            self._parse_html,
            self._remove_urls,
            self._remove_punctuation,
            self._remove_numbers,
            self._remove_numbers2,
            self._remove_non_ascii,
            self._remove_lone_punct,
            self._remove_ws,
            self._turn_lower,
        ]

    def _setup_replacement_dict(
        self, acronym_fp: str = "../private_data/Acronym List 2025.xlsx"
    ):
        if acronym_fp:
            # load in acronyms
            acronyms_df = pd.read_excel(acronym_fp)
            acronyms_df = acronyms_df.dropna(subset=["Full Name or Definition"])

            # convert into a dictionary of acronym: replacement
            acronym_conversions = dict(
                zip(acronyms_df["Acronym"], acronyms_df["Full Name or Definition"])
            )
            # only keep acronyms that are longer than one character
            self._acronym_conversions = {
                key: value
                for key, value in acronym_conversions.items()
                if len(key) > 1 or key == "IT"
            }
            print("created")

            del self._acronym_conversions["IT"]
            del self._acronym_conversions["MFX"]
            del self._acronym_conversions["KB"]
            del self._acronym_conversions["STA"]
            del self._acronym_conversions["AIR"]
            del self._acronym_conversions["MRCO"]

    def process_string(self, text: str, replace_abbrv: bool = False) -> str:
        """Perform preprocessing on a text string.

        Args:
            text (str): Input string

        Returns:
            str: Preprocessed string
        """
        # print(text)
        if replace_abbrv and self._acronym_conversions:
            self._functions.insert(8, self._replace_acronyms)

        return self._clean(text)

    def process_file(self, path: Path) -> str:
        """Perform preprocessing on text from a file.

        Args:
            text (str): File to read

        Returns:
            str: Preprocessed file contents
        """
        text = path.open().read()
        return self._clean(text)

    def _clean(self, text):
        for clean_fn in self._functions:
            _log.debug(f"clean text with {clean_fn.__name__}")
            text = clean_fn(text)
        return text

    def _remove_newlines(self, text):
        """Remove newlines, tabs, and carriage returns"""
        return re.sub(r"[\n\t\r]", " ", text)

    def _remove_urls(self, text):
        """Remove URLs"""
        url_string = r"\S*https?://\S*"
        url_replacement = " "
        no_url_txt = re.sub(url_string, url_replacement, text)

        url_string = r"\S*www\.\S*"
        url_replacement = " "
        no_url_txt = re.sub(url_string, url_replacement, no_url_txt)
        return no_url_txt

    def _parse_html(self, text):
        """Remove HTML tags"""
        no_html_text = BeautifulSoup(text, "lxml").text
        return no_html_text

    def _remove_punctuation(self, text):
        """Remove Punctuation"""
        #  Define punctuation to remove (excluding ., -, :, ;)
        # punct = '!"#$%&\'()*+/<=>?@[\\]^_`{|}~:;,•’”“\/'
        # =============================
        remove = punctuation + "•" + "–" + "’" + "”" + "“"
        remove = remove.replace(".", "")  # don't remove full-stops
        remove = remove.replace(",", "")  # don't remove commas
        remove = remove.replace("’", "")  # don't remove apostrophes
        remove = remove.replace("'", "")  # don't remove apostrophes
        remove = remove.replace("_", "")  # don't remove apostrophes

        remove = remove.replace(
            "-", ""
        )  # don't blanket-remove hyphens, will be removed after
        # remove = remove.replace(":", "")  # don't remove colons
        # remove = remove.replace(";", "")  # don't remove semi-colons
        punct_pattern = r"[{}]".format(remove)  # create the pattern
        punct_pattern = r"[{}]".format(remove)  # create the pattern
        no_punct_txt = re.sub(punct_pattern, " ", text)

        # Only remove dashes with spaces before or after them. This way abbreviation replacements with dashes are preserved.
        # Remove ellipsis (...)
        no_punct_txt = re.sub(
            r"(\s?)\.{2,}",
            r".",
            re.sub(r"(\s?)[-](\s)", r" ", re.sub(r"(\s)[-](\s?)", r" ", no_punct_txt)),
        )
        return no_punct_txt

    def _remove_numbers(self, text):
        """Remove numbers."""
        # Remove all decimal numbers
        number_pattern = r"(^|\s)\d+(?:\.\d+)?"  # \s\d+'
        number_replacement = " "  # ' nos'
        no_digit_txt = re.sub(number_pattern, number_replacement, text)
        return no_digit_txt

    def _remove_numbers2(self, text):
        """Remove all numbers preceeded by a non-alphanumeric character,
        e.g. a dash. This will leave chemical formulas in.
        """
        number_pattern = r"[^a-zA-Z0-9]+\d+"
        number_replacement = " "  #
        return re.sub(number_pattern, number_replacement, text)

    def _remove_non_ascii(self, text):
        """Remove non-ascii characters."""
        return text.encode("ascii", "ignore").decode()

    def _remove_lone_punct(self, text):
        """Remove lone punctuation."""
        return re.sub(r"(?<=\s)[^\w\s]+(?=\s)", "", text)

    def _remove_ws(self, text):
        """Remove extra whitespaces."""
        whitespace_pattern = r"\s+"
        no_whitespace_txt = re.sub(whitespace_pattern, " ", text)
        # no_whitespace_txt = re.sub(r"[\n\t\r]", "", no_whitespace_txt)
        no_whitespace_txt = no_whitespace_txt.rstrip()
        no_whitespace_txt = no_whitespace_txt.lstrip()
        return no_whitespace_txt

    def _turn_lower(self, text):
        """Turn into lowercase"""
        return text.lower()

    def _replace_acronyms(self, text: str) -> str:
        """Perform replacement of acryonyms to their full name

        Args:
            text (str): Input string
            fp (str): Pointer to list of acronyms

        Returns:
            str: Preprocessed string
        """
        patterns = []
        for acronym in self._acronym_conversions.keys():
            patterns.extend([acronym.upper(), acronym.lower()])
        patterns = sorted(self._acronym_conversions.keys(), key=len, reverse=True)
        pattern = (
            r"(?<![a-zA-Z])(?:"
            + "|".join(re.escape(acronym) for acronym in patterns)
            + r")(?![a-z])"
        )
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        # compiled_pattern = re.compile(pattern)
        res_str = re.sub(compiled_pattern, self._replace_acronym, text)

        return res_str

    def _replace_acronym(self, acronym):
        return self._acronym_conversions.get(acronym.group().upper(), acronym.group())

    def get_abbrv(self, text):
        patterns = []
        for acronym in self._acronym_conversions.keys():
            patterns.extend([acronym.upper(), acronym.lower()])
        patterns = sorted(self._acronym_conversions.keys(), key=len, reverse=True)

        pattern = (
            r"(?<![a-zA-Z])(?:"
            + "|".join(re.escape(acronym) for acronym in patterns)
            + r")(?![a-z])"
        )
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        all_matches = re.findall(compiled_pattern, text)
        c = Counter()
        c.update(all_matches)
        return c
