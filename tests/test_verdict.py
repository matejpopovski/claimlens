import numpy as np
import pytest
from transformers import AutoTokenizer

from claimlens.calibration import softmax
from claimlens.data import LABELS, NEI, REFUTES, SUPPORTS, Abstract
from claimlens.encoders import CrossEncoder
from claimlens.verdict import (
    AbstractJudgement,
    NLIModel,
    format_premise,
    judge_abstracts,
    label_order,
    marginalise_evidence,
    pool_claim_logits,
    score_sentences,
    select_rationale,
    with_nei_weight,
)


def test_label_order_handles_either_model_convention():
    assert label_order({0: "entailment", 1: "neutral", 2: "contradiction"}) == [0, 2, 1]
    assert label_order({0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}) == [1, 0, 2]


def test_label_order_refuses_an_unknown_head():
    with pytest.raises(KeyError):
        label_order({0: "positive", 1: "negative", 2: "neutral"})


def test_select_rationale_orders_by_score_and_caps_the_count():
    scores = np.array([0.1, 0.9, 0.6, 0.95, 0.7])
    assert select_rationale(scores, threshold=0.5, max_sentences=3) == (3, 1, 4)


def test_select_rationale_falls_back_to_the_best_sentence():
    assert select_rationale(np.array([0.1, 0.2]), threshold=0.5, max_sentences=3) == (1,)


def test_select_rationale_breaks_ties_by_position():
    assert select_rationale(np.array([0.5, 0.5]), threshold=0.1, max_sentences=1) == (0,)


def test_pooling_lets_one_abstract_with_evidence_settle_the_claim():
    probabilities = np.array([[0.02, 0.03, 0.95], [0.90, 0.05, 0.05]])
    assert LABELS[pool_claim_logits(probabilities).argmax()] == SUPPORTS


def test_pooling_all_silent_abstracts_gives_not_enough_info():
    probabilities = np.array([[0.05, 0.05, 0.9], [0.04, 0.06, 0.9]])
    assert LABELS[pool_claim_logits(probabilities).argmax()] == NEI


def test_nei_bias_can_overturn_a_weak_decisive_signal():
    probabilities = np.array([[0.4, 0.3, 0.3]])
    assert LABELS[pool_claim_logits(probabilities).argmax()] == SUPPORTS
    assert LABELS[pool_claim_logits(probabilities, nei_bias=1.0).argmax()] == NEI


def test_pooling_nothing_retrieved_gives_not_enough_info():
    assert LABELS[pool_claim_logits(np.zeros((0, 3))).argmax()] == NEI


def test_pooling_a_single_abstract_preserves_its_ranking():
    probabilities = np.array([[0.1, 0.7, 0.2]])
    assert LABELS[pool_claim_logits(probabilities).argmax()] == REFUTES


def test_format_premise_puts_the_title_first():
    assert format_premise("Title.", ["One.", "Two."]) == "Title. One. Two."


def test_judgement_exposes_label_and_evidence_scores():
    judgement = AbstractJudgement(1, (0.1, 0.7, 0.2), (0,), (0.6, 0.1))
    assert judgement.label == REFUTES
    assert judgement.is_decisive


def test_marginalising_moves_doubt_about_evidence_into_nei():
    combined = marginalise_evidence(np.array([0.8, 0.1, 0.1]), 0.5)
    assert combined == pytest.approx([0.4, 0.05, 0.55])
    assert combined.sum() == pytest.approx(1.0)


def test_certain_evidence_leaves_the_nli_answer_unchanged():
    assert marginalise_evidence(np.array([0.2, 0.7, 0.1]), 1.0) == pytest.approx([0.2, 0.7, 0.1])


def test_zero_nei_weight_leaves_the_nli_only_choosing_a_direction():
    combined = marginalise_evidence(np.array([0.3, 0.1, 0.6]), 0.8, nei_weight=0.0)
    assert combined == pytest.approx([0.6, 0.2, 0.2])


def test_with_nei_weight_matches_a_fresh_combination():
    judgement = AbstractJudgement(
        1, (0.0, 0.0, 1.0), (1,), (0.1, 0.8), nli_probabilities=(0.3, 0.1, 0.6)
    )
    updated = with_nei_weight(judgement, 0.5)
    expected = marginalise_evidence(np.array([0.3, 0.1, 0.6]), 0.8, nei_weight=0.5)
    assert updated.probabilities == pytest.approx(tuple(expected))
    assert judgement.probabilities == (0.0, 0.0, 1.0)


def test_no_evidence_means_not_enough_info_whatever_the_nli_says():
    assert marginalise_evidence(np.array([0.9, 0.05, 0.05]), 0.0) == pytest.approx([0, 0, 1])


def test_nli_logits_follow_project_label_order(tiny_nli_dir):
    model = NLIModel.load(tiny_nli_dir, device="cpu")
    raw = model.model.config.id2label
    assert raw[0] == "entailment"
    assert raw[2] == "contradiction"
    tokens = model.tokenizer("aspirin", "fever", return_tensors="pt")
    native = model.model(**tokens).logits[0].detach().numpy()
    ours = model.logits([("aspirin", "fever")])[0]
    assert ours == pytest.approx(native[[0, 2, 1]], abs=1e-5)


def test_nli_logits_keep_input_order_despite_length_batching(tiny_nli_dir):
    model = NLIModel.load(tiny_nli_dir, device="cpu", batch_size=2)
    pairs = [("aspirin", "fever"), ("a study of a trial", "patients"), ("vitamin", "scurvy")]
    together = model.logits(pairs)
    one_by_one = np.vstack([model.logits([pair]) for pair in pairs])
    assert together == pytest.approx(one_by_one, abs=1e-4)


def test_selector_and_nli_end_to_end_on_tiny_models(tiny_nli_dir, tiny_reranker_dir):
    selector = CrossEncoder.load(tiny_reranker_dir, device="cpu")
    model = NLIModel.load(tiny_nli_dir, device="cpu")
    abstract = Abstract(5, "aspirin", ("aspirin reduces fever", "a study", "no effect"))
    requests = [("aspirin reduces fever", abstract)]
    evidence = score_sentences(selector, requests)
    assert evidence[0].shape == (3,)
    assert ((evidence[0] >= 0) & (evidence[0] <= 1)).all()

    judged = judge_abstracts(model, requests, evidence, threshold=0.0, max_sentences=2)
    assert len(judged[0].rationale) == 2
    assert sum(judged[0].probabilities) == pytest.approx(1.0)
    assert judged[0].sentence_evidence == pytest.approx(tuple(evidence[0]))


def test_selector_reads_the_sentence_without_the_title():
    class RecordingSelector:
        pairs = None

        def score_pairs(self, pairs):
            RecordingSelector.pairs = pairs
            return np.zeros(len(pairs))

    abstract = Abstract(5, "A very matching title", ("One.", "Two.", "Three."))
    scores = score_sentences(RecordingSelector(), [("claim", abstract)])
    assert RecordingSelector.pairs[0] == ("claim", "One.")
    assert scores[0] == pytest.approx([0.5, 0.5, 0.5])


def test_nli_reads_the_rationale_in_document_order():
    class RecordingModel:
        pairs = None

        def logits(self, pairs):
            RecordingModel.pairs = pairs
            return np.zeros((len(pairs), 3))

    abstract = Abstract(5, "Title.", ("First.", "Second.", "Third.", "Fourth."))
    evidence = np.array([0.2, 0.9, 0.2, 0.8])
    judged = judge_abstracts(
        RecordingModel(), [("claim", abstract)], [evidence], threshold=0.5, max_sentences=2
    )
    assert judged[0].rationale == (1, 3), "strongest first, for the official score"
    assert RecordingModel.pairs == [("Title. Second. Fourth.", "claim")]
    expected = marginalise_evidence(softmax(np.zeros(3)), 0.9)
    assert judged[0].probabilities == pytest.approx(tuple(expected))


def test_empty_inputs_do_not_reach_the_models(tiny_nli_dir, tiny_reranker_dir):
    selector = CrossEncoder.load(tiny_reranker_dir, device="cpu")
    model = NLIModel.load(tiny_nli_dir, device="cpu")
    assert score_sentences(selector, []) == []
    assert judge_abstracts(model, [], [], threshold=0.5, max_sentences=3) == []


def test_softmax_of_pooled_scores_is_a_distribution():
    pooled = pool_claim_logits(np.array([[0.2, 0.3, 0.5]]))
    assert softmax(pooled).sum() == pytest.approx(1.0)


def test_tokenizer_fixture_is_a_real_fast_tokenizer(tiny_nli_dir):
    assert AutoTokenizer.from_pretrained(tiny_nli_dir).is_fast
