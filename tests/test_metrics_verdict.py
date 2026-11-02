import pytest
from sklearn.metrics import f1_score

from claimlens.data import LABELS, NEI, REFUTES, SUPPORTS, Claim, Rationale
from claimlens.metrics_verdict import (
    AbstractPrediction,
    classification_report,
    scifact_scores,
)


def worked_example():
    """The scoring example from doc/evaluation.md in allenai/scifact."""
    claim = Claim(
        52,
        "ALDH1 expression is associated with poorer prognosis.",
        (11, 15),
        {
            11: (Rationale((0, 1), SUPPORTS), Rationale((11,), SUPPORTS)),
            15: (Rationale((4,), SUPPORTS),),
        },
    )
    predictions = {
        52: {
            11: AbstractPrediction(SUPPORTS, (1, 11, 13)),
            16: AbstractPrediction(REFUTES, (18, 20)),
        }
    }
    return claim, predictions


def test_official_worked_example_abstract_level():
    claim, predictions = worked_example()
    scores = scifact_scores([claim], predictions)
    assert scores["abstract_label_only"]["precision"] == pytest.approx(1 / 2)
    assert scores["abstract_label_only"]["recall"] == pytest.approx(1 / 2)
    assert scores["abstract_label_rationale"]["f1"] == pytest.approx(1 / 2)


def test_official_worked_example_sentence_level():
    claim, predictions = worked_example()
    scores = scifact_scores([claim], predictions)
    assert scores["sentence_selection"]["precision"] == pytest.approx(1 / 5)
    assert scores["sentence_selection"]["recall"] == pytest.approx(1 / 4)
    assert scores["sentence_selection"]["f1"] == pytest.approx(2 / 9)


def test_wrong_label_keeps_selection_credit_but_loses_label_credit():
    claim = Claim(1, "c", (5,), {5: (Rationale((2,), SUPPORTS),)})
    scores = scifact_scores([claim], {1: {5: AbstractPrediction(REFUTES, (2,))}})
    assert scores["sentence_selection"]["f1"] == 1.0
    assert scores["sentence_selection_label"]["f1"] == 0.0
    assert scores["abstract_label_only"]["f1"] == 0.0


def test_abstract_scoring_only_reads_the_first_three_sentences():
    claim = Claim(1, "c", (5,), {5: (Rationale((9,), SUPPORTS),)})
    late = {1: {5: AbstractPrediction(SUPPORTS, (1, 2, 3, 9))}}
    early = {1: {5: AbstractPrediction(SUPPORTS, (9, 1, 2, 3))}}
    assert scifact_scores([claim], late)["abstract_label_rationale"]["f1"] == 0.0
    assert scifact_scores([claim], early)["abstract_label_rationale"]["f1"] == 1.0


def test_abstract_scoring_extends_past_three_for_long_gold_rationales():
    claim = Claim(1, "c", (5,), {5: (Rationale((1, 2, 3, 4), SUPPORTS),)})
    full = {1: {5: AbstractPrediction(SUPPORTS, (1, 2, 3, 4))}}
    assert scifact_scores([claim], full)["abstract_label_rationale"]["f1"] == 1.0


def test_no_predictions_scores_zero_without_dividing_by_zero():
    claim = Claim(1, "c", (5,), {5: (Rationale((1,), SUPPORTS),)})
    assert scifact_scores([claim], {})["abstract_label_only"]["f1"] == 0.0


def test_nei_predictions_cannot_be_scored_as_abstracts():
    with pytest.raises(ValueError, match="decisive"):
        AbstractPrediction(NEI, ())


def test_macro_f1_agrees_with_sklearn():
    gold = [SUPPORTS, SUPPORTS, REFUTES, NEI, NEI, REFUTES, SUPPORTS]
    predicted = [SUPPORTS, REFUTES, REFUTES, NEI, SUPPORTS, NEI, SUPPORTS]
    report = classification_report(gold, predicted)
    assert report["macro_f1"] == pytest.approx(f1_score(gold, predicted, average="macro"))
    assert report["accuracy"] == pytest.approx(4 / 7)
    assert set(report["per_class"]) == set(LABELS)


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="length"):
        classification_report([SUPPORTS], [])
