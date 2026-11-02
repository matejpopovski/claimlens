"""Checks that need real weights and data on disk. Run with: pytest -m models

These are skipped by default and in CI, because CI downloads no models. They exist to
catch what tiny random models cannot: a wrong label mapping or a swapped premise.
"""

from pathlib import Path

import pytest

from claimlens.data import LABELS, NEI, REFUTES, SUPPORTS, DataPaths, load_corpus
from claimlens.encoders import CrossEncoder, TextEncoder
from claimlens.models import MODELS, local_path
from claimlens.verdict import NLIModel

pytestmark = pytest.mark.models


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} is not on disk")
    return path


@pytest.fixture(scope="module")
def stock_nli():
    return NLIModel.load(require(local_path("deberta-nli")), device="cpu")


@pytest.mark.parametrize(
    ("premise", "hypothesis", "expected"),
    [
        ("Aspirin lowered fever in all treated adults.", "Aspirin lowers fever.", SUPPORTS),
        ("Aspirin had no effect on fever in adults.", "Aspirin lowers fever.", REFUTES),
        ("The trial enrolled forty adults in Ohio.", "Aspirin lowers fever.", NEI),
    ],
)
def test_stock_nli_label_mapping_matches_its_meaning(stock_nli, premise, hypothesis, expected):
    logits = stock_nli.logits([(premise, hypothesis)])
    assert LABELS[int(logits.argmax())] == expected


def test_premise_and_hypothesis_are_not_swapped(stock_nli):
    # Entailment is directional: the specific premise entails the general hypothesis.
    specific = "Daily aspirin lowered fever in elderly women."
    general = "Aspirin lowers fever."
    forward = stock_nli.logits([(specific, general)])[0]
    backward = stock_nli.logits([(general, specific)])[0]
    assert forward[LABELS.index(SUPPORTS)] > backward[LABELS.index(SUPPORTS)]


def test_real_encoder_prefers_the_related_passage():
    spec = MODELS["bge-small"]
    encoder = TextEncoder(
        require(local_path("bge-small")),
        pooling=spec.pooling,
        query_prefix=spec.query_prefix,
        device="cpu",
    )
    documents = encoder.encode(["Aspirin reduces fever in adults.", "Whales migrate each year."])
    query = encoder.encode(["Does aspirin lower a fever?"], is_query=True)[0]
    assert documents[0] @ query > documents[1] @ query


def test_real_reranker_prefers_the_related_passage():
    reranker = CrossEncoder.load(require(local_path("ms-marco-minilm")), device="cpu")
    scores = reranker.score(
        "Does aspirin lower a fever?",
        ["Aspirin reduces fever in adults.", "Whales migrate each year."],
    )
    assert scores[0] > scores[1]


def test_corpus_is_the_published_size():
    corpus = load_corpus(require(DataPaths(Path("data")).corpus))
    assert len(corpus) == 5183
