"""The whole path on tiny random models: structure and policy, not model quality."""

from dataclasses import replace

import numpy as np
import pytest

from claimlens.abstain import AbstentionPolicy
from claimlens.data import LABELS, NEI
from claimlens.encoders import CrossEncoder, TextEncoder
from claimlens.fusion import reciprocal_rank_fusion
from claimlens.pipeline import ClaimVerifier
from claimlens.retrieval import Hit, HybridRetriever, RetrievalConfig, best_rerank_score
from claimlens.verdict import NLIModel, VerdictConfig


@pytest.fixture
def retriever(corpus, tiny_encoder_dir, tiny_reranker_dir):
    encoder = TextEncoder(tiny_encoder_dir, device="cpu")
    embeddings = encoder.encode([a.full_text for a in corpus.values()])
    return HybridRetriever(
        corpus,
        RetrievalConfig(fusion_depth=3, rerank_depth=3),
        encoder=encoder,
        embeddings=embeddings,
        reranker=CrossEncoder.load(tiny_reranker_dir, device="cpu"),
    )


@pytest.fixture
def verifier(retriever, tiny_nli_dir, tiny_reranker_dir):
    def make(policy, top_k=2):
        return ClaimVerifier(
            retriever,
            CrossEncoder.load(tiny_reranker_dir, device="cpu"),
            NLIModel.load(tiny_nli_dir, device="cpu"),
            verdict_config=VerdictConfig(top_k=top_k, rationale_threshold=0.2),
            policy=policy,
        )

    return make


def test_search_merges_both_lists_and_keeps_the_trail(retriever):
    hits = retriever.search("aspirin reduces fever", 3)
    assert sorted(h.doc_id for h in hits) == [11, 12, 13]
    assert next(h for h in hits if h.doc_id == 11).bm25_rank == 1
    assert all(h.dense_rank is not None for h in hits)
    scores = [h.rerank_score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k(retriever):
    assert len(retriever.search("aspirin", 1)) == 1


def test_bm25_ranking_uses_the_hand_written_index(retriever):
    assert retriever.bm25_ranking("scurvy") == [12]
    assert retriever.bm25_ranking("nothingmatches") == []


def test_dense_weight_reaches_the_fusion(corpus, tiny_encoder_dir, tiny_reranker_dir):
    encoder = TextEncoder(tiny_encoder_dir, device="cpu")
    weighted = HybridRetriever(
        corpus,
        RetrievalConfig(fusion_depth=3, rerank_depth=3, rrf_k=10, dense_weight=4.0),
        encoder=encoder,
        embeddings=encoder.encode([a.full_text for a in corpus.values()]),
        reranker=CrossEncoder.load(tiny_reranker_dir, device="cpu"),
    )
    hit = next(h for h in weighted.search("aspirin reduces fever", 3) if h.doc_id == 11)
    expected = 1 / (10 + hit.bm25_rank) + 4.0 / (10 + hit.dense_rank)
    assert hit.rrf_score == pytest.approx(expected)


def test_final_fusion_can_restore_the_first_stage_order(
    corpus, tiny_encoder_dir, tiny_reranker_dir
):
    encoder = TextEncoder(tiny_encoder_dir, device="cpu")

    config = RetrievalConfig(fusion_depth=3, rerank_depth=3)
    claim = "aspirin reduces fever"

    def build(**options):
        return HybridRetriever(
            corpus,
            replace(config, **options),
            encoder=encoder,
            embeddings=encoder.encode([a.full_text for a in corpus.values()]),
            reranker=CrossEncoder.load(tiny_reranker_dir, device="cpu"),
        )

    def search(**options):
        return build(**options).search(claim, 3)

    lexical = build().bm25_ranking(claim)
    semantic = build().dense_ranking(claim)

    plain = search()
    assert [h.rerank_score for h in plain] == sorted((h.rerank_score for h in plain), reverse=True)
    restored = search(final_fusion_k=1, final_fusion_weight=1000.0)
    expected = reciprocal_rank_fusion(
        [lexical, semantic], k=config.rrf_k, weights=[1.0, config.dense_weight]
    )
    assert [h.doc_id for h in restored] == [doc_id for doc_id, _ in expected]
    assert {h.rerank_score for h in restored} == {h.rerank_score for h in plain}


def test_best_rerank_score_ignores_list_position():
    hits = [Hit(1, 0.5, 0.1, 1, 1), Hit(2, 3.0, 0.1, 2, 2)]
    assert best_rerank_score(hits) == 3.0
    assert best_rerank_score([]) is None


def test_verify_returns_a_complete_result(verifier):
    result = verifier(AbstentionPolicy(min_confidence=0.0)).verify("aspirin reduces fever")
    assert not result.decision.abstained
    assert result.decision.verdict in LABELS
    assert set(result.decision.probabilities) == set(LABELS)
    assert sum(result.decision.probabilities.values()) == pytest.approx(1.0)
    assert len(result.evidence) == 2
    for item in result.evidence:
        assert item.judgement.doc_id == item.hit.doc_id == item.abstract.doc_id
        assert len(item.judgement.sentence_evidence) == len(item.abstract.sentences)
        assert item.judgement.rationale
    assert result.seconds > 0


def test_an_impossible_threshold_makes_the_system_abstain(verifier):
    result = verifier(AbstentionPolicy(min_confidence=1.01)).verify("aspirin reduces fever")
    assert result.decision.abstained
    assert result.decision.verdict is None
    assert result.decision.abstentions[0].reason.value == "low_confidence"


def test_retrieval_floor_is_read_from_the_top_hit(verifier):
    result = verifier(AbstentionPolicy(min_confidence=0.0, min_retrieval_score=1e6)).verify(
        "aspirin reduces fever"
    )
    assert [a.reason.value for a in result.decision.abstentions] == ["weak_retrieval"]


def test_temperature_softens_without_changing_the_leading_label(
    retriever, tiny_nli_dir, tiny_reranker_dir
):
    def run(temperature):
        verifier = ClaimVerifier(
            retriever,
            CrossEncoder.load(tiny_reranker_dir, device="cpu"),
            NLIModel.load(tiny_nli_dir, device="cpu"),
            verdict_config=VerdictConfig(top_k=2, temperature=temperature),
            policy=AbstentionPolicy(min_confidence=0.0),
        )
        return verifier.verify("aspirin reduces fever").decision

    cold, warm = run(1.0), run(5.0)
    assert cold.best_guess == warm.best_guess
    assert warm.confidence < cold.confidence
    assert np.isfinite(warm.confidence)


def test_nei_weight_reaches_the_verdict(retriever, tiny_nli_dir, tiny_reranker_dir):
    verifier = ClaimVerifier(
        retriever,
        CrossEncoder.load(tiny_reranker_dir, device="cpu"),
        NLIModel.load(tiny_nli_dir, device="cpu"),
        verdict_config=VerdictConfig(top_k=1, nli_nei_weight=0.0),
        policy=AbstentionPolicy(min_confidence=0.0),
    )
    judgement = verifier.verify("aspirin reduces fever").evidence[0].judgement
    evidence = judgement.sentence_evidence[judgement.rationale[0]]
    # With the NLI model's NEI output weighted to zero, doubt comes from the selector alone.
    assert judgement.probabilities[LABELS.index(NEI)] == pytest.approx(1.0 - evidence)
