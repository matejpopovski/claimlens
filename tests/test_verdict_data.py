import pytest

from claimlens.data import NEI, REFUTES, SUPPORTS, Abstract, Claim, Rationale
from claimlens.verdict_data import (
    balanced_class_weights,
    build_nli_examples,
    build_selector_examples,
)


@pytest.fixture
def abstract():
    return Abstract(
        1,
        "Title",
        (
            "Aspirin reduces fever in adults.",
            "Sample size was forty patients.",
            "Aspirin was given daily for fever.",
            "Funding came from a charity.",
            "Fever fell after aspirin in most adults.",
        ),
    )


def supporting_claim():
    return Claim(1, "Aspirin reduces fever.", (1,), {1: (Rationale((0,), SUPPORTS),)})


def test_gold_rationale_is_the_positive_example(abstract):
    examples = build_nli_examples([supporting_claim()], {1: abstract}, {})
    positives = [e for e in examples if e.source == "rationale"]
    assert len(positives) == 1
    assert positives[0].label == SUPPORTS
    assert "Aspirin reduces fever in adults." in positives[0].premise


def test_rationale_sentences_never_appear_in_negatives(abstract):
    examples = build_nli_examples([supporting_claim()], {1: abstract}, {}, nei_per_decisive=10.0)
    negatives = [e for e in examples if e.label == NEI]
    assert negatives
    assert all("Aspirin reduces fever in adults." not in e.premise for e in negatives)


def test_negatives_prefer_sentences_that_look_relevant(abstract):
    examples = build_nli_examples([supporting_claim()], {1: abstract}, {}, nei_per_decisive=10.0)
    singles = [e for e in examples if e.source == "evidence_abstract_sentence"]
    assert singles
    assert all("Funding" not in e.premise for e in singles)


def test_a_distractor_is_added_to_each_rationale_with_the_same_label(abstract):
    claim = Claim(1, "Aspirin reduces fever.", (1,), {1: (Rationale((0,), REFUTES),)})
    examples = build_nli_examples([claim], {1: abstract}, {})
    extended = [e for e in examples if e.source == "rationale_plus_distractor"]
    plain = next(e for e in examples if e.source == "rationale")
    assert len(extended) == 1
    assert extended[0].label == REFUTES
    assert "Aspirin reduces fever in adults." in extended[0].premise
    assert len(extended[0].premise) > len(plain.premise)


def test_negatives_are_capped_relative_to_decisive_examples(abstract):
    claims = [
        Claim(i, f"Aspirin claim {i}.", (1,), {1: (Rationale((0,), SUPPORTS),)}) for i in range(4)
    ]
    examples = build_nli_examples(claims, {1: abstract}, {}, nei_per_decisive=0.5)
    assert sum(e.label != NEI for e in examples) == 8
    assert sum(e.label == NEI for e in examples) == 4


def test_cited_abstract_without_evidence_gives_only_negatives(abstract):
    no_evidence = Claim(1, "Aspirin reduces fever.", (1,), {})
    decisive = Claim(2, "Other.", (2,), {2: (Rationale((0,), SUPPORTS),)})
    corpus = {1: abstract, 2: Abstract(2, "T", ("One.", "Two.", "Three."))}
    examples = build_nli_examples([no_evidence, decisive], corpus, {}, nei_per_decisive=10.0)
    cited = [e for e in examples if e.source.startswith("cited_no_evidence")]
    assert cited
    assert {e.label for e in cited} == {NEI}


def test_retrieved_abstracts_skip_cited_ones(abstract):
    other = Abstract(2, "Other", ("Aspirin in the elderly.", "Two.", "Three."))
    examples = build_nli_examples(
        [supporting_claim()], {1: abstract, 2: other}, {1: [1, 2]}, nei_per_decisive=20.0
    )
    retrieved = [e for e in examples if e.source.startswith("retrieved")]
    assert retrieved
    assert all(e.premise.startswith("Other") for e in retrieved)


def test_build_is_deterministic(abstract):
    first = build_nli_examples([supporting_claim()], {1: abstract}, {})
    assert first == build_nli_examples([supporting_claim()], {1: abstract}, {})


def test_balanced_weights_favour_the_rare_class(abstract):
    claims = [
        Claim(1, "a", (1,), {1: (Rationale((0,), SUPPORTS),)}),
        Claim(2, "b", (1,), {1: (Rationale((2,), REFUTES),)}),
    ]
    examples = build_nli_examples(claims, {1: abstract}, {}, nei_per_decisive=2.0)
    support, refute, nei = balanced_class_weights(examples)
    assert support == refute > nei
    assert (support + refute + nei) / 3 == pytest.approx(1.0)


def test_weights_refuse_a_missing_class(abstract):
    examples = build_nli_examples([supporting_claim()], {1: abstract}, {})
    with pytest.raises(ValueError, match="no examples"):
        balanced_class_weights(examples)


def test_selector_examples_label_only_rationale_sentences_positive(abstract):
    claim = Claim(1, "Aspirin reduces fever.", (1,), {1: (Rationale((0, 2), SUPPORTS),)})
    examples = build_selector_examples([claim], {1: abstract}, {})
    assert len(examples) == len(abstract.sentences)
    positive = {sentence for _, sentence, label in examples if label == 1.0}
    assert positive == {abstract.sentences[0], abstract.sentences[2]}


def test_selector_examples_treat_uncited_retrieved_abstracts_as_negative(abstract):
    other = Abstract(2, "Other", ("Aspirin in the elderly.", "Two.", "Three.", "Four."))
    examples = build_selector_examples(
        [supporting_claim()],
        {1: abstract, 2: other},
        {1: [1, 2]},
        sentences_per_retrieved_abstract=2,
    )
    from_other = [e for e in examples if e[1] in other.sentences]
    assert len(from_other) == 2
    assert "Aspirin in the elderly." in {e[1] for e in from_other}
    assert all(label == 0.0 for _, _, label in from_other)
