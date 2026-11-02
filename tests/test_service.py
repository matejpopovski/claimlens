"""API tests on tiny random models, so outcomes are forced through the policy."""

import pytest
from fastapi.testclient import TestClient

from claimlens.abstain import AbstentionPolicy
from claimlens.encoders import CrossEncoder, TextEncoder
from claimlens.pipeline import ClaimVerifier
from claimlens.retrieval import HybridRetriever, RetrievalConfig
from claimlens.service.app import Example, create_app
from claimlens.verdict import NLIModel, VerdictConfig


@pytest.fixture
def make_client(corpus, tiny_encoder_dir, tiny_reranker_dir, tiny_nli_dir):
    encoder = TextEncoder(tiny_encoder_dir, device="cpu")
    retriever = HybridRetriever(
        corpus,
        RetrievalConfig(fusion_depth=3, rerank_depth=3),
        encoder=encoder,
        embeddings=encoder.encode([a.full_text for a in corpus.values()]),
        reranker=CrossEncoder.load(tiny_reranker_dir, device="cpu"),
    )

    def make(policy, **options):
        verifier = ClaimVerifier(
            retriever,
            CrossEncoder.load(tiny_reranker_dir, device="cpu"),
            NLIModel.load(tiny_nli_dir, device="cpu"),
            verdict_config=VerdictConfig(top_k=2, temperature=1.5),
            policy=policy,
        )
        return TestClient(create_app(verifier, weights="tiny-test", **options))

    return make


def test_verify_answers_with_sentences_and_a_retrieval_trail(make_client):
    client = make_client(AbstentionPolicy(min_confidence=0.0))
    response = client.post("/api/verify", json={"claim": " aspirin reduces fever "})
    assert response.status_code == 200
    body = response.json()
    assert body["claim"] == "aspirin reduces fever"
    assert body["outcome"] in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
    assert body["abstentions"] == []
    paper = body["evidence"][0]
    assert paper["rank"] == 1
    assert {"rerank_score", "rrf_score", "bm25_rank", "dense_rank"} == set(paper["retrieval"])
    assert any(s["rationale_rank"] == 1 for s in paper["sentences"])
    assert all(0.0 <= s["evidence_score"] <= 1.0 for s in paper["sentences"])


def test_abstention_reports_the_reason_and_keeps_the_guess_separate(make_client):
    client = make_client(AbstentionPolicy(min_confidence=1.01))
    body = client.post("/api/verify", json={"claim": "aspirin reduces fever"}).json()
    assert body["outcome"] == "ABSTAINED"
    assert body["abstentions"][0]["reason"] == "low_confidence"
    assert "1.01" in body["abstentions"][0]["message"]
    assert body["best_guess"] in ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO")
    assert body["evidence"], "an abstention still shows what was read"


@pytest.mark.parametrize("claim", ["", "ab", "x" * 501])
def test_claim_length_is_validated(make_client, claim):
    client = make_client(AbstentionPolicy(min_confidence=0.0))
    assert client.post("/api/verify", json={"claim": claim}).status_code == 422


def test_health_describes_weights_and_policy(make_client):
    client = make_client(
        AbstentionPolicy(min_confidence=0.7, min_retrieval_score=1.0),
        weights_note="stock weights",
    )
    body = client.get("/api/health").json()
    assert body["corpus_size"] == 3
    assert body["weights"] == "tiny-test"
    assert body["weights_note"] == "stock weights"
    assert body["temperature"] == 1.5
    assert body["policy"]["min_retrieval_score"] == 1.0


def test_examples_are_served(make_client):
    client = make_client(
        AbstentionPolicy(min_confidence=0.0), examples=[Example(claim="c", source="s")]
    )
    assert client.get("/api/examples").json() == [{"claim": "c", "source": "s"}]


def test_reader_page_and_assets_are_served(make_client):
    client = make_client(AbstentionPolicy(min_confidence=0.0))
    page = client.get("/")
    assert page.status_code == 200
    assert 'id="claim-form"' in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_page_script_builds_text_nodes_only(make_client):
    # Abstract text comes from a dataset, so it must never reach innerHTML.
    client = make_client(AbstentionPolicy(min_confidence=0.0))
    assert "innerHTML" not in client.get("/static/app.js").text
