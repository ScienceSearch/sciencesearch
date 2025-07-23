import pytest
from sciencesearch.nlp.models import (
    CaseSensitiveStemmer,
    NullStemmer,
    Stopwords,
    Parameters,
    Algorithm,
    Ensemble,
    Rake,
    Yake,
    KPMiner,
    PS,
    reduce_duplicates,
    remove_substrings,
)


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


class FakeAlg(Algorithm):
    "Override abstract method to allow instantiation"

    def _get_keywords(self, text):
        return ["foo"]


@pytest.mark.unit
def test_algorithm_base():
    alg = FakeAlg()
    alg = FakeAlg(num_keywords=5)
    # type checking on/off
    with pytest.raises(TypeError):
        alg = FakeAlg(num_keywords="5")
    alg = FakeAlg(num_keywords="5", check_types=False)
    # methods:
    alg = FakeAlg(num_keywords=5)
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
    print(kw)
    assert len(kw) == n
    print(f"Keywords: {kw}")
    kw_lower = [x.lower() for x in kw]
    assert "gregor samsa" in kw_lower


@pytest.mark.unit
def test_rake():
    n = 5
    rk = Rake(num_keywords=n)
    kw = rk.run(kafka_text)
    print(kw)

    assert len(kw) == n
    print(f"Keywords: {kw}")
    kw_lower = [x.lower() for x in kw]
    assert "troubled dreams" in kw_lower


@pytest.mark.unit
def test_multi_algorithm():
    yk1 = Yake(num_keywords=5)
    yk2 = Yake(num_keywords=10)
    rk1 = Rake(num_keywords=5)
    rk2 = Rake(num_keywords=10)
    ma = Ensemble(yk1, yk2, rk1)
    ma.add(rk2)
    ma_kw = ma.run(kafka_text)
    expect_kw = list(
        set(yk1.run(kafka_text))
        .union(set(yk2.run(kafka_text)))
        .union(set(rk1.run(kafka_text)))
        .union(set(rk2.run(kafka_text)))
    )
    reduce1 = reduce_duplicates(expect_kw)
    reduce2 = remove_substrings(reduce1)
    assert sorted(ma_kw) == sorted(reduce2)
