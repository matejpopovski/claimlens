import pytest

from claimlens.text import s_stem, tokenize


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("cells", "cell"),
        ("therapies", "therapy"),
        ("diseases", "disease"),
        ("virus", "virus"),
        ("mass", "mass"),
        ("does", "doe"),
        ("gas", "gas"),
    ],
)
def test_s_stem_handles_plurals_without_touching_latin_endings(word, stem):
    assert s_stem(word) == stem


def test_greek_symbols_match_their_spelled_names():
    assert tokenize("TNF-α") == tokenize("TNF-alpha") == ["tnf", "alpha"]


def test_alphanumeric_gene_names_survive():
    assert tokenize("p53 and IL-6 levels") == ["p53", "il", "6", "level"]


def test_stopword_removal_is_optional():
    assert tokenize("the cell", remove_stopwords=False, stem=False) == ["the", "cell"]
    assert tokenize("the cell") == ["cell"]
