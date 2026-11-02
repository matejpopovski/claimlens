import numpy as np
import pytest

from claimlens.abstain import AbstentionPolicy
from claimlens.data import LABELS, NEI, REFUTES, SUPPORTS, Abstract, Claim, Rationale
from claimlens.encoders import CrossEncoder
from claimlens.evaluation import (
    abstract_predictions,
    apply_policy,
    claim_logits,
    claim_report,
    confidence_and_correct,
    gold_indices,
    load_hits,
    run_sentence_pass,
    save_hits,
    selective_summary,
    sentence_idf,
    tfidf_top_sentences,
)
from claimlens.retrieval import Hit
from claimlens.verdict import AbstractJudgement, NLIModel

CLAIMS = [
    Claim(1, "a", (5,), {5: (Rationale((0,), SUPPORTS),)}),
    Claim(2, "b", (6,), {6: (Rationale((1,), REFUTES),)}),
    Claim(3, "c", (7,), {}),
]


def judgement(doc_id, probabilities, rationale=(0,)):
    return AbstractJudgement(doc_id, probabilities, rationale, (0.4,))


JUDGED = {
    1: [judgement(5, (0.9, 0.05, 0.05))],
    2: [judgement(6, (0.1, 0.8, 0.1), (1,))],
    3: [judgement(7, (0.1, 0.1, 0.8))],
}


def test_hits_round_trip_through_disk(tmp_path):
    hits = {1: [Hit(5, 1.5, 0.1, 2, None)]}
    save_hits(tmp_path / "nested" / "hits.json", hits)
    assert load_hits(tmp_path / "nested" / "hits.json") == hits


def test_claim_logits_and_report():
    logits = claim_logits(CLAIMS, JUDGED, top_k=3, nei_bias=0.0)
    assert logits.shape == (3, 3)
    report = claim_report(CLAIMS, logits)
    assert report["macro_f1"] == 1.0
    assert (gold_indices(CLAIMS) == logits.argmax(axis=1)).all()


def test_claim_with_no_retrieved_abstract_is_not_enough_info():
    logits = claim_logits(CLAIMS, {}, top_k=3, nei_bias=0.0)
    assert [LABELS[i] for i in logits.argmax(axis=1)] == [NEI, NEI, NEI]


def test_top_k_limits_the_abstracts_pooled():
    judged = {1: [judgement(5, (0.1, 0.1, 0.8)), judgement(9, (0.9, 0.05, 0.05))]}
    one = claim_logits(CLAIMS[:1], judged, top_k=1, nei_bias=0.0)
    two = claim_logits(CLAIMS[:1], judged, top_k=2, nei_bias=0.0)
    assert LABELS[one[0].argmax()] == NEI
    assert LABELS[two[0].argmax()] == SUPPORTS


def test_abstract_predictions_follow_the_claim_verdict():
    judged = {
        1: [judgement(5, (0.9, 0.05, 0.05)), judgement(8, (0.1, 0.8, 0.1))],
        3: [judgement(7, (0.1, 0.1, 0.8))],
    }
    logits = np.log(np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]]))
    predictions = abstract_predictions(CLAIMS, judged, logits, top_k=3)
    assert set(predictions) == {1}
    assert set(predictions[1]) == {5}


def test_confidence_and_correct():
    logits = claim_logits(CLAIMS, JUDGED, top_k=3, nei_bias=0.0)
    confidence, correct = confidence_and_correct(CLAIMS, logits, temperature=2.0)
    assert correct.all()
    assert confidence.min() < confidence_and_correct(CLAIMS, logits)[0].max()


def test_policy_and_selective_summary():
    logits = claim_logits(CLAIMS, JUDGED, top_k=3, nei_bias=0.0)
    probabilities_top = np.exp(logits).max(axis=1)
    threshold = float(np.sort(probabilities_top)[1])
    decisions = apply_policy(
        CLAIMS,
        JUDGED,
        {},
        logits,
        top_k=3,
        temperature=1.0,
        policy=AbstentionPolicy(threshold + 0.01),
    )
    summary = selective_summary(CLAIMS, decisions)
    assert summary["answered"] == sum(not d.abstained for d in decisions)
    assert summary["coverage"] == pytest.approx(summary["answered"] / 3)
    assert summary["selective_accuracy"] == 1.0


def test_policy_reads_the_best_reranker_score_among_the_abstracts_used():
    logits = claim_logits(CLAIMS[:1], JUDGED, top_k=1, nei_bias=0.0)
    strong_second = {1: [Hit(5, 0.1, 0.1, 1, 1), Hit(9, 9.0, 0.1, 2, 2)]}

    def abstained(top_k, floor):
        return apply_policy(
            CLAIMS[:1],
            JUDGED,
            strong_second,
            logits,
            top_k=top_k,
            temperature=1.0,
            policy=AbstentionPolicy(0.0, min_retrieval_score=floor),
        )[0].abstained

    assert abstained(top_k=1, floor=5.0), "a hit outside top_k must not count"
    assert not abstained(top_k=2, floor=5.0)


def test_selective_summary_counts_decisive_answers_apart_from_nei():
    logits = claim_logits(CLAIMS, JUDGED, top_k=3, nei_bias=0.0)
    decisions = apply_policy(
        CLAIMS, JUDGED, {}, logits, top_k=3, temperature=1.0, policy=AbstentionPolicy(0.0)
    )
    summary = selective_summary(CLAIMS, decisions)
    assert summary["decisive_answers"] == 2
    assert summary["decisive_precision"] == 1.0


def test_selective_summary_with_nothing_answered():
    logits = claim_logits(CLAIMS, JUDGED, top_k=3, nei_bias=0.0)
    decisions = apply_policy(
        CLAIMS, JUDGED, {}, logits, top_k=3, temperature=1.0, policy=AbstentionPolicy(1.01)
    )
    assert selective_summary(CLAIMS, decisions)["selective_accuracy"] == 0.0


def test_sentence_pass_groups_judgements_by_claim(tiny_nli_dir, tiny_reranker_dir):
    corpus = {5: Abstract(5, "aspirin", ("aspirin reduces fever", "a study"))}
    nli = NLIModel.load(tiny_nli_dir, device="cpu")
    selector = CrossEncoder.load(tiny_reranker_dir, device="cpu")
    result = run_sentence_pass(selector, CLAIMS[:2], {1: [5], 2: [5]}, corpus)
    judged = result.judge(nli, threshold=0.1, max_sentences=2)
    assert set(judged) == {1, 2}
    assert judged[1][0].doc_id == 5


def test_tfidf_prefers_the_sentence_sharing_rare_terms():
    abstract = Abstract(
        1, "t", ("Patients were enrolled in a study.", "Aspirin reduces fever.", "Study ended.")
    )
    idf = sentence_idf({1: abstract})
    assert tfidf_top_sentences("does aspirin reduce fever", abstract, idf, 1) == [1]
