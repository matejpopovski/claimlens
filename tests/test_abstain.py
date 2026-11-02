import numpy as np
import pytest

from claimlens.abstain import (
    AbstentionPolicy,
    Reason,
    confidence_features,
    decide,
    has_conflict,
    threshold_for_precision,
)
from claimlens.data import NEI, REFUTES, SUPPORTS

NONE = np.zeros((0, 3))


def test_confident_answer_passes_every_check():
    decision = decide(
        [0.9, 0.05, 0.05],
        top_retrieval_score=5.0,
        abstract_probabilities=np.array([[0.9, 0.05, 0.05]]),
        policy=AbstentionPolicy(0.7, 1.0, 0.5),
    )
    assert decision.verdict == SUPPORTS
    assert not decision.abstained
    assert decision.confidence == pytest.approx(0.9)


def test_low_confidence_abstains_but_keeps_the_best_guess_for_display():
    decision = decide(
        [0.5, 0.3, 0.2],
        top_retrieval_score=None,
        abstract_probabilities=NONE,
        policy=AbstentionPolicy(0.7),
    )
    assert decision.abstained
    assert decision.verdict is None
    assert decision.best_guess == SUPPORTS
    assert decision.abstentions[0].reason is Reason.LOW_CONFIDENCE
    assert "0.50" in decision.abstentions[0].message
    assert "chance of being correct" in decision.abstentions[0].message


def test_not_enough_info_is_an_answer_not_an_abstention():
    decision = decide(
        [0.05, 0.05, 0.9],
        top_retrieval_score=4.0,
        abstract_probabilities=np.array([[0.05, 0.05, 0.9]]),
        policy=AbstentionPolicy(0.7, 1.0, 0.5),
    )
    assert decision.verdict == NEI
    assert not decision.abstained


def test_weak_retrieval_overrides_a_confident_verdict():
    decision = decide(
        [0.05, 0.05, 0.9],
        top_retrieval_score=-3.0,
        abstract_probabilities=np.array([[0.05, 0.05, 0.9]]),
        policy=AbstentionPolicy(0.7, 0.0, 0.5),
    )
    assert [a.reason for a in decision.abstentions] == [Reason.WEAK_RETRIEVAL]


def test_no_retrieved_abstract_counts_as_weak_retrieval():
    decision = decide(
        [0.0, 0.0, 1.0],
        top_retrieval_score=None,
        abstract_probabilities=NONE,
        policy=AbstentionPolicy(0.7, 0.0),
    )
    assert decision.abstentions[0].reason is Reason.WEAK_RETRIEVAL
    assert "no abstract" in decision.abstentions[0].message


def test_conflicting_abstracts_abstain_even_when_pooled_confidence_is_high():
    decision = decide(
        [0.8, 0.15, 0.05],
        top_retrieval_score=5.0,
        abstract_probabilities=np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]),
        policy=AbstentionPolicy(0.7, 1.0, 0.5),
    )
    assert [a.reason for a in decision.abstentions] == [Reason.CONFLICTING_EVIDENCE]


def test_every_failed_check_is_reported():
    decision = decide(
        [0.4, 0.4, 0.2],
        top_retrieval_score=-1.0,
        abstract_probabilities=np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2]]),
        policy=AbstentionPolicy(0.7, 0.0, 0.5),
    )
    assert {a.reason for a in decision.abstentions} == set(Reason)


def test_disabled_checks_do_not_fire():
    decision = decide(
        [0.9, 0.05, 0.05],
        top_retrieval_score=-100.0,
        abstract_probabilities=np.array([[0.9, 0.0, 0.1], [0.0, 0.9, 0.1]]),
        policy=AbstentionPolicy(0.5, None, None),
    )
    assert not decision.abstained


def test_conflict_needs_both_directions():
    agreeing = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    assert not has_conflict(agreeing, 0.5)
    assert not has_conflict(NONE, 0.5)


def test_threshold_for_precision_hand_computed():
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, False])
    assert threshold_for_precision(confidences, correct, target=0.65) == 0.7


def test_threshold_takes_the_largest_coverage_that_meets_the_target():
    # An early dip must not hide the fact that wider coverage still meets the target.
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([True, False, True, True, True])
    assert threshold_for_precision(confidences, correct, target=0.75) == 0.5


def test_a_lucky_handful_is_not_enough():
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([True, True, False, False, False])
    assert threshold_for_precision(confidences, correct, target=0.9) == 0.8
    assert threshold_for_precision(confidences, correct, target=0.9, min_answered=3) is None


def test_unreachable_target_returns_none():
    assert threshold_for_precision(np.array([0.9]), np.array([False]), target=0.5) is None


def test_tied_confidences_are_never_split():
    confidences = np.array([0.9, 0.9, 0.5])
    correct = np.array([True, False, False])
    assert threshold_for_precision(confidences, correct, target=0.9) is None


def test_confidence_features_describe_the_top_prediction():
    features = confidence_features(np.array([[0.75, 0.2, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]))
    assert features[0] == pytest.approx([np.log(3.0), 0.0, 0.0])
    assert features[1, 1:] == pytest.approx([1.0, 0.0])
    assert features[2, 1:] == pytest.approx([0.0, 1.0])


def test_confidence_model_can_rate_refutes_below_supports_at_the_same_probability():
    policy = AbstentionPolicy(0.5, confidence_weights=(1.0, -1.0, 0.0, 0.0))
    probabilities = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    supports, refutes = policy.confidence(probabilities)
    assert refutes < supports
    assert supports == pytest.approx(0.8)


def test_without_a_confidence_model_the_top_probability_is_used():
    policy = AbstentionPolicy(0.5)
    assert policy.confidence(np.array([[0.8, 0.1, 0.1]])) == pytest.approx([0.8])


def test_confidence_of_a_single_row_is_a_scalar():
    policy = AbstentionPolicy(0.5, confidence_weights=(1.0, 0.0, 0.0, 0.0))
    assert np.ndim(policy.confidence(np.array([0.8, 0.1, 0.1]))) == 0


def test_decide_compares_the_estimated_chance_with_the_threshold():
    policy = AbstentionPolicy(0.5, confidence_weights=(1.0, -5.0, 0.0, 0.0))
    probabilities = [0.1, 0.8, 0.1]
    decision = decide(
        probabilities,
        top_retrieval_score=None,
        abstract_probabilities=NONE,
        policy=policy,
    )
    assert decision.abstained
    assert decision.best_guess == REFUTES
    assert decision.confidence < 0.5
