"""
Natural Language Processing models to extract keywords from documents
"""

# stdlib
from abc import ABC, abstractmethod
from collections import namedtuple
import logging
import itertools
from operator import itemgetter
from pathlib import Path
from string import punctuation
import time
from typing import Iterable, Optional, Type, Any

# third-party
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import yake
from rake_nltk import Rake as _Rake
from rake_nltk import Metric
import pke
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import similarities
from sklearn.metrics.pairwise import cosine_similarity

__this_dir = Path(__file__).parent  # guess
DATA_DIR = __this_dir.parent.parent / "data"  # could be off
DEFAULTS = {"stopwords_file": DATA_DIR / "stopwords_en.txt"}

_log = logging.getLogger(__name__)


class CaseSensitiveStemmer(PorterStemmer):
    """Case-sensitive stemming"""

    def stem(self, word: str) -> str:
        """Stem the input.

        Args:
            word: Input word.

        Returns:
            str: Stem of input word.
        """
        word_stem = super().stem(word)
        if word == word.title():
            word_stem = word_stem.capitalize()
        elif word.isupper():
            word_stem = word_stem.upper()
        return word_stem


class NullStemmer(PorterStemmer):
    """Null pattern for stemming."""

    def stem(self, word: str) -> str:
        "Stemmeth not, lest ye be stemmed."
        return word


def stem(text):
    stemmer = CaseSensitiveStemmer()
    words = word_tokenize(text)
    stemmed_words = map(stemmer.stem, words)
    return " ".join(stemmed_words)


class Stopwords:
    """
    Creates a list of stopwords, words to ignore.

    Attributes:
        sw: A set of stopwords.
        stemmer: Instance of a stemming algorithm.
    
    """
    ENCODING = "utf-8"

    def __init__(self, stemmer: PorterStemmer = None):
        """Inits Stopwords class with a list of stopwords and a stemming algorithm."""
        self._sw = set()  # stopword list, initially empty
        self._stemmer = stemmer or NullStemmer()

    @property
    def stopwords(self):
        return list(self._sw)

    def add_file(self, input_file: str | Path) -> int:
        """Add all stopwords in input file to the stopword list.

        Args:
            input_file: The input file

        Return:
            Number of stopwords added
        """
        file_path = input_file if isinstance(input_file, Path) else Path(input_file)
        with open(file_path, "r", encoding=self.ENCODING) as f:
            n = 0
            for line in f:
                word = line.strip().lower()
                self._sw.add(self._stemmer.stem(word))
                n += 1
        return n

    def add_list(self, words: Iterable[str]) -> int:
        """Add all stopwords in input list to the stopword list.

        Args:
            words: An iterable of stopwords

        Return:
            Number of stopwords added
        """
        for word in words:
            self._sw.add(self._stemmer.stem(word))
        return len(words)


class Parameters:
    """ Creates a set of parameters an NLP models.
     
    Attributes:
        spec: List of named tuples that define the 
                (specification, (parameter name, parameter type, description, and default value)
                for each paramter required to run a keyword extraction algorithm.
        values: A dictionary of the paramater names and their values.
        check_types: A boolean indicating if types should be checked.

    
    """

    def __init__(self, spec: list[tuple], values: dict[str, Any], check_types=True):
        """
        Inits Parameters class with a list of specification definitions and a dictionary of 
            defined specification and values.
        
        """
        self._spec = spec
        # set all values to defaults
        self._v = {x.name: x.default for x in spec}
        v_types = {x.name: x.type for x in spec}
        # fill in specified values
        for k, v in values.items():
            try:
                if check_types:
                    if not isinstance(v, v_types[k]):
                        raise TypeError(
                            f"Value for parameter '{k}' ({v}), is not type {v_types[k]}"
                        )
                self._v[k] = v
            except KeyError:
                me = self.__class__.__name__
                raise ValueError(f"Unknown parameter '{k}' for {me}")

    def __getattr__(self, name):
        """Access parameters as attributes."""
        if name in self._v:
            return self._v[name]
        raise KeyError(f"Cannot get value for unknown parameter '{name}'")

    def str(self):
        plist = [f"{k}={v}" for k, v in self._v.items()]
        return ", ".join(plist)


#: Parameter specification type
PS = namedtuple("Spec", ("name", "type", "desc", "default"))


class Algorithm(ABC):
    """Abstract base class of keyword extraction algorithms.

    Sublasses should:
      - define their own PARAM_SPEC as a concatenation of the
        base PARAM_SPEC and their specific parameters
      - call `super().__init__(**kwargs)` in their constructor
      - override the method `_get_keywords()` to return keywords for
        a given text


    Attributes:
        check_types: Whether to check parameter types
        params: Parameter values
    """

    # constants for keyword_sort param
    _S_OCC, _S_SCR = "occ", "score"

    # common scale range for scores and counts
    _SCORE_RANGE = 100

    PARAM_SPEC = [
        PS("stopwords", Stopwords, "Stopwords", None),
        PS("stemming", bool, "Whether to do stemming", False),
        PS("num_keywords", int, "How many keywords to extract", 10),
        PS(
            "keyword_sort",
            list,
            f"sort orderings: {_S_OCC} (number of occurrences), {_S_SCR}, "
            f"or a dict with weights for each of these keys, e.g., "
            f"{{'{_S_OCC}': 0.75, '{_S_SCR}': 0.25}}, "
            f"and additionally a flag 'i' for ignoring keyword case",
            [],
        ),
    ]

    def __init__(self, check_types: bool = True, **params):
        """Inits a base constructor for Algorithm with choice to check types and parameter values"""
        self.params = Parameters(self.PARAM_SPEC, params, check_types=check_types)
        if self.params.stopwords is None:
            stemmer = CaseSensitiveStemmer() if self.params.stemming else NullStemmer()
            self.params.stopwords = Stopwords(stemmer=stemmer)
            sw_file = DEFAULTS["stopwords_file"]
            try:
                self.params.stopwords.add_file(sw_file)
            except IOError as err:
                raise ValueError(
                    f"Could not load stopwords from default file '{sw_file}': {err}"
                )
        self._run_timings = {}
        self._name = self.__class__.__name__

    @classmethod
    def get_params(cls):
        return cls.PARAM_SPEC

    @classmethod
    def print_params(cls):
        num_base = len(Algorithm.PARAM_SPEC)
        print("Common:")
        for i, p in enumerate(cls.PARAM_SPEC):
            if i == num_base:
                print(f"{cls.__name__}:")
            print(f"  - {p.type.__name__} {p.name}: {p.desc}. Default is {p.default}")

    def run(self, text: str) -> list[str]:
        _log.info(f"Run algorithm: {self._name}")
        _log.debug(f"Run with parameters: {str(self.params)}")
        t0 = time.time()
        text = self._stem_text(text)
        t1 = time.time() - t0
        kw = self._get_keywords(text)
        t2 = time.time() - t1
        self._run_timings = {"stem": t1, "extract": t2, "total": t1 + t2}
        if not kw:
            # stop and catch fire if no keywords (looking at you, KPMiner)
            raise RuntimeError("No keywords extracted")
        _log.debug(
            f"Finished algorithm {self._name} time={self._run_timings['total']:.3g}s"
        )
        return kw

    @abstractmethod
    def _get_keywords(self, text: str) -> list[str]:
        pass

    def _stem_text(self, text: str) -> str:
        if not self.params.stemming:
            return text
        return stem(text)

    @property
    def timings(self):
        if not self._run_timings:
            raise ValueError("No timings before algorithm is run")
        return self._run_timings[:]  # copy, avoid mods

    def _sort(
        self,
        keywords: list[str],
        scores: list[float | int],
        text: str,
        higher_is_better=True,
    ) -> list[str]:
        """Sort keywords by one or more criteria.

        TODO: Generalize this for Yake and KPMiner

        Args:
            keywords: List of keywords (same length as scores)
            scores: List of numeric scores
            text (str): Input text
            higher_is_better: If True, higher scores are better

        Raises:
            ValueError: Bad sort method or invalid input for the
                        dict of key/weight values for the weighted sort.

        Returns:
            list[str]: Keywords, sorted
        """
        n = len(keywords)
        # scale the scores, flipping high/low to force higher to be better
        scaled_scores = self._scale_scores(scores, flip=not higher_is_better)
        # (keyword, (sort_key1, sort_key2, ..))
        kw_key = [(keywords[i], []) for i in range(n)]
        counts, first_count = {}, True
        if "i" in self.params.keyword_sort:
            ignore_case = True
        else:
            ignore_case = False
        # add one new sort key for each method
        for sort_method in self.params.keyword_sort:
            if isinstance(sort_method, dict):
                try:
                    score_weight = float(sort_method[self._S_SCR])
                    occ_weight = float(sort_method[self._S_OCC])
                except (ValueError, KeyError):
                    raise ValueError(
                        f"'keyword_order' dict parameter must have "
                        f"two keys, '{self._S_SCR}' and '{self._S_OCC}', with the value "
                        f"for each key being the numeric weight"
                    )
                # get counts if needed
                if first_count:
                    if ignore_case:
                        lower_text = text.lower()
                    for k, _ in kw_key:
                        txt, case_k = (
                            (lower_text, k.lower()) if ignore_case else (text, k)
                        )
                        count = self._count_occ(txt, case_k)
                        counts[case_k] = count
                    first_count = False
                # calculate scaling for counts
                min_count, max_count = 1e12, -1
                for k, _ in kw_key:
                    case_k = k.lower() if ignore_case else k
                    value = counts[case_k]
                    min_count = min(min_count, value)
                    max_count = max(max_count, value)
                count_delta = max_count - min_count
                assert count_delta > 0
                count_offs = -min_count
                count_factor = self._SCORE_RANGE / count_delta
                # calculate scaled values
                for i, (k, keys) in enumerate(kw_key):
                    case_k = k.lower() if ignore_case else k
                    scaled_count = (counts[case_k] + count_offs) * count_factor
                    value = scaled_scores[i] * score_weight + scaled_count * occ_weight
                    keys.append(value)
            elif sort_method == self._S_SCR:
                for i, (k, keys) in enumerate(kw_key):
                    keys.append(scaled_scores[i])
            elif sort_method == self._S_OCC:
                if first_count and ignore_case:
                    lower_text = text.lower()
                for k, keys in kw_key:
                    if first_count:
                        txt, case_k = (
                            (lower_text, k.lower()) if ignore_case else (text, k)
                        )
                        count = self._count_occ(txt, case_k)
                        counts[case_k] = count
                    else:
                        case_k = k.lower() if ignore_case else k
                        count = counts[case_k]
                    keys.append(count)
                first_count = False
            elif sort_method == "i":
                pass
            else:
                raise ValueError(
                    f"Unknown 'keyword_order' method: {sort_method}. "
                    f"Must be '{self._S_SCR}', '{self._S_OCC}', or a dict"
                )
        # sort by the compound key, higher is better
        kw_key.sort(key=itemgetter(1), reverse=True)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"raw sorted keywords: {kw_key}")
        # return just the keywords
        return (item[0] for item in kw_key)

    @staticmethod
    def _count_occ(txt, s):
        return txt.count(s)

    def _scale_scores(self, scores, flip: bool = False):
        min_score = min(scores)
        max_score = max(scores)
        if min_score == max_score:
            return scores.copy()
        delta = max_score - min_score
        offset = -min_score
        factor = self._SCORE_RANGE / delta
        if flip:
            result = [self._SCORE_RANGE - (v + offset) * factor for v in scores]
        else:
            result = [(v + offset) * factor for v in scores]
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"_scale_scores: raw={scores} ; scaled={result}")
        return result


## Algorithms


class KPMiner(Algorithm):
    """
    An instance of a KP Miner NLP Model.      
    """

    PARAM_SPEC = Algorithm.PARAM_SPEC + [
        PS("lasf", int, "Last allowable seen frequency", 3),
        PS(
            "cutoff",
            int,
            "Cutoff threshold for number of words after which "
            "if a phrase appears for the first time it is ignored",
            400,
        ),
        PS(
            "alpha",
            float,
            "Weight-adjustment parameter 1 for boosting factor."
            "See original paper for definition",
            2.3,
        ),
        PS(
            "sigma",
            float,
            "Weight-adjustment parameter 2 for boosting factor."
            "See original paper for definition",
            3.0,
        ),
        PS(
            "doc_freq_info",
            object,
            "Document frequency counts. "
            "Default (None) uses the semeval2010 counts"
            "provided in 'df-semeval2010.tsv.gz'",
            None,
        ),
    ]

    def __init__(self, **kwargs):
        """Inits KPMiner with an information to initialize super class Algorithm and model instance"""
        super().__init__(**kwargs)
        self._extractor = pke.unsupervised.KPMiner()

    def _get_keywords(self, text):
        stopwords = self.params.stopwords.stopwords
        self._extractor.load_document(input=text, language="en", stoplist=stopwords)
        self._extractor.candidate_selection(
            lasf=self.params.lasf, cutoff=self.params.cutoff
        )
        self._extractor.candidate_weighting(
            df=self.params.doc_freq_info,
            sigma=self.params.sigma,
            alpha=self.params.alpha,
        )
        kw_score = self._extractor.get_n_best(
            n=self.params.num_keywords, stemming=self.params.stemming
        )
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"Raw KPMiner result: {kw_score}")
        # separate keywords from scores
        keywords = list(map(itemgetter(0), kw_score))
        # optionally sort by provided criteria
        if self.params.keyword_sort:
            scores = list(map(itemgetter(1), kw_score))
            kw = list(self._sort(keywords, scores, text))
        else:
            kw = list(keywords)
        return kw


class Rake(Algorithm):
    """
    An instance of a RAKE NLP Model.      
    """
     
    PARAM_SPEC = Algorithm.PARAM_SPEC + [
        PS("min_len", int, "Minimum ngram size", 1),
        PS("max_len", int, "Maximum ngram size", 3),
        PS(
            "min_kw_len",
            int,
            "Minimum keyword length. Applied as post-processing filter.",
            3,
        ),
        PS(
            "min_kw_occ",
            int,
            "Mimumum number of occurences of keyword in text string."
            "Applied as post-processing filter.",
            4,
        ),
        PS(
            "ranking_metric",
            Any,
            "ranking parameter for rake algorithm",
            Metric.DEGREE_TO_FREQUENCY_RATIO,
        ),
        PS(
            "include_repeated_phrases",
            bool,
            "boolean for determining whether multiple of the same keywords "
            "are output by rake",
            True,
        ),
    ]

    def __init__(self, **kwargs):
        """Inits RAKW with an information to initialize super class Algorithm and model instance"""
        super().__init__(**kwargs)
        self._extractor = _Rake(
            stopwords=self.params.stopwords.stopwords,
            min_length=self.params.min_len,
            max_length=self.params.max_len,
            include_repeated_phrases=self.params.include_repeated_phrases,
            ranking_metric=self.params.ranking_metric,
        )

    def _get_keywords(self, text: str) -> list[str]:
        self._extractor.extract_keywords_from_text(text)
        score_kw = self._extractor.get_ranked_phrases_with_scores()
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"Raw Rake result: {score_kw}")
        # remove keywords that are too short
        score_kw = [item for item in score_kw if len(item[1]) >= self.params.min_kw_len]
        # separate keywords from scores
        keywords = list(map(itemgetter(1), score_kw))
        # optionally sort by provided criteria
        if self.params.keyword_sort:
            scores = list(map(itemgetter(0), score_kw))
            kw = list(self._sort(keywords, scores, text))
        else:
            kw = list(keywords)
        return kw[: self.params.num_keywords]


class Yake(Algorithm):
    """
    An instance of a YAKE NLP Model.      
    """
        
    PARAM_SPEC = Algorithm.PARAM_SPEC + [
        PS("ws", int, "YAKE window size", 2),
        PS("dedup", float, "Deduplication limit for YAKE", 0.9),
        PS("dedup_method", str, "method ('leve', 'seqm' or 'jaro')", "leve"),
        PS("ngram", int, "Maximum ngram size", 2),
    ]

    def __init__(self, **kwargs):
        """Inits YAKE with an information to initialize super class Algorithm and model instance"""
        super().__init__(**kwargs)

        # Initialize Yake
        self._extractor = yake.KeywordExtractor(
            top=self.params.num_keywords,
            n=self.params.ngram,
            dedupLim=self.params.dedup,
            dedupFunc=self.params.dedup_method,
            windowsSize=self.params.ws,
            stopwords=self.params.stopwords.stopwords,
        )

    def _get_keywords(self, text: str) -> list[str]:
        """Get YAKE keywords

        Args:
            text (str): Input text

        Returns:
            list[str]: List of keywords
        """
        kw_score = self._extractor.extract_keywords(text)
        if _log.isEnabledFor(logging.DEBUG):
            _log.debug(f"Raw Yake result: {kw_score}")
        # separate keywords from scores
        keywords = list(map(itemgetter(0), kw_score))
        # optionally sort by provided criteria
        if self.params.keyword_sort:
            scores = list(map(itemgetter(1), kw_score))
            kw = list(self._sort(keywords, scores, text))
        else:
            kw = list(keywords)
        return kw


class Ensemble(Algorithm):
    def __init__(self, *algs, **kwargs):
        if "stopwords" not in kwargs:
            # don't add default stopwords
            kwargs["stopwords"] = []
        super().__init__(check_types=False, **kwargs)
        self._algorithms = list(algs)

    def add(self, alg: Algorithm):
        self._algorithms.append(alg)

    def _get_keywords(self, text):
        merged_kw = set()
        scores = {}
        for alg in self._algorithms:
            keywords = alg.run(text)
            for kw in keywords:
                merged_kw.add(kw)
        return list(merged_kw)
