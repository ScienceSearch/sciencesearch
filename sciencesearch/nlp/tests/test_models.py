import pytest
from sciencesearch.nlp.models import (
    CaseSensitiveStemmer,
    NullStemmer,
    Stopwords,
    Parameters,
    Algorithm,
    Rake,
    Yake,
    KPMiner,
    PS,
)
from sciencesearch.nlp.sweep import Sweep


@pytest.mark.unit
def test_casesensitive_stemmer():
    st = CaseSensitiveStemmer()
    assert st.stem("leaving") == "leav"
    assert st.stem("Leaving") == "Leav"
    assert st.stem("LEAVING") == "LEAV"


@pytest.mark.unit
def test_null_stemmer():
    st = NullStemmer()
    assert st.stem("leaving") == "leaving"


SW_LIST = ["dr", "dra", "mr", "ms", "a", "a's", "able", "about", "above", "according"]


@pytest.mark.unit
def test_stopwords(tmp_path):
    data_path = tmp_path / "stopwords.txt"
    sw = Stopwords()
    # start empty
    assert sw.stopwords == []
    # add from list
    sw.add_list(SW_LIST)
    assert set(sw.stopwords) == set(SW_LIST)
    # ignore dup
    sw.add_list(SW_LIST)
    assert set(sw.stopwords) == set(SW_LIST)
    # add from file
    sw = Stopwords()
    with open(data_path, "w", encoding=Stopwords.ENCODING) as f:
        for word in SW_LIST:
            f.write(word)
            f.write("\n")
    n = sw.add_file(data_path)
    assert n == len(SW_LIST)
    assert set(sw.stopwords) == set(SW_LIST)


P_SPEC = [
    PS("stopwords", Stopwords, "Stopwords", None),
    PS("stemming", bool, "Whether to do stemming", False),
]


class Foowords(Stopwords):
    pass


@pytest.mark.unit
def test_parameters():
    p = Parameters(P_SPEC, {})
    # bad attr
    with pytest.raises(KeyError):
        _ = p.foo
    # default value
    assert p.stopwords == P_SPEC[0][-1]
    # user value
    sw = Stopwords()
    sw.add_list(SW_LIST)
    p = Parameters(P_SPEC, {"stopwords": sw})
    assert p.stopwords == sw
    # bad type
    sw = ["what", "wot", "wat"]
    with pytest.raises(TypeError):
        p = Parameters(P_SPEC, {"stopwords": sw})
    # subclass OK
    sw = Foowords()
    p = Parameters(P_SPEC, {"stopwords": sw})


class TestAlg(Algorithm):
    "Override abstract method to allow instantiation"

    def _get_keywords(self, text):
        return ["foo"]


@pytest.mark.unit
def test_algorithm_base():
    alg = TestAlg()
    alg = TestAlg(num_keywords=5)
    # type checking on/off
    with pytest.raises(TypeError):
        alg = TestAlg(num_keywords="5")
    alg = TestAlg(num_keywords="5", check_types=False)
    # methods:
    alg = TestAlg(num_keywords=5)
    assert alg.get_params() != {}
    alg.print_params()
    kw = alg.run("Hello, world")
    assert kw == ["foo"]


kafka_text = """
One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed
in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted
his head a little he could see his brown belly, slightly domed and divided by arches
into stiff sections. The bedding was hardly able to cover it and seemed ready to 
slide off any moment. His many legs, pitifully thin compared with the size of the 
rest of him, waved about helplessly as he looked. "What's happened to me?" he 
thought. It wasn't a dream. His room, a proper human
"""


@pytest.mark.unit
def test_yake():
    n = 5
    yk = Yake(num_keywords=n)
    kw = yk.run(kafka_text)
    assert len(kw) == n
    print(f"Keywords: {kw}")
    kw_lower = [x.lower() for x in kw]
    assert "gregor samsa" in kw_lower


@pytest.mark.unit
def test_kpminer():
    n = 5
    kp = KPMiner(num_keywords=n)
    text = kafka_text * 10
    kw = kp.run(text)
    assert len(kw) == n
    print(f"Keywords: {kw}")
    kw_lower = [x.lower() for x in kw]
    assert "gregor samsa" in kw_lower


@pytest.mark.unit
def test_rake():
    n = 5
    rk = Rake(num_keywords=n)
    kw = rk.run(kafka_text)
    assert len(kw) == n
    print(f"Keywords: {kw}")
    kw_lower = [x.lower() for x in kw]
    assert "troubled dreams" in kw_lower


@pytest.mark.unit
def test_sweep():
    sweep = Sweep(Yake)
    # bad param for range
    with pytest.raises(KeyError):
        sweep.set_param_range("foo", 0, 10, step=1)
    # range
    sweep.set_param_range("dedup", 0.85, 0.95, step=0.05)
    sweep.set_param_range("ngram", lb=1, ub=3, nsteps=2)
    r = sweep.run(kafka_text)
    assert len(r.results) == 9  # 3x3 run
