"""Build training examples for the rationale selector and the NLI model from SciFact.

The negatives take the care, because both models only ever meet sentences that already
look relevant. Random sentences would teach them almost nothing, so negatives are chosen
by overlap with the claim, from non-rationale sentences of evidence abstracts, cited
abstracts with no evidence, and highly retrieved abstracts the claim never cited.

The two sets differ in balance. The selector sees every sentence of an abstract, as it
will at inference. The NLI model gets far fewer negatives, because the selector has
already screened what it reads, and flooding it with near misses is what made the earlier
one-model design call true rationales NOT_ENOUGH_INFO.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from claimlens.data import LABELS, NEI, Abstract, Claim
from claimlens.text import tokenize
from claimlens.verdict import format_premise


@dataclass(frozen=True)
class VerdictExample:
    premise: str
    claim: str
    label: str
    source: str


def _overlap(claim_tokens: set[str], sentence: str) -> float:
    tokens = set(tokenize(sentence))
    union = claim_tokens | tokens
    return len(claim_tokens & tokens) / len(union) if union else 0.0


def _ranked_sentences(claim_tokens: set[str], abstract: Abstract, exclude: set[int]) -> list[int]:
    candidates = [i for i in range(len(abstract.sentences)) if i not in exclude]
    return sorted(candidates, key=lambda i: (-_overlap(claim_tokens, abstract.sentences[i]), i))


def build_nli_examples(
    claims: Sequence[Claim],
    corpus: Mapping[int, Abstract],
    retrieved: Mapping[int, Sequence[int]],
    *,
    negatives_per_abstract: int = 2,
    retrieved_abstracts: int = 3,
    nei_per_decisive: float = 1.0,
    seed: int = 13,
) -> list[VerdictExample]:
    decisive: list[VerdictExample] = []
    negatives: list[VerdictExample] = []

    def example(claim, abstract, indices, label, source) -> VerdictExample:
        sentences = [abstract.sentences[i] for i in sorted(indices)]
        return VerdictExample(format_premise(abstract.title, sentences), claim.text, label, source)

    for claim in claims:
        claim_tokens = set(tokenize(claim.text))

        for doc_id, rationales in claim.evidence.items():
            abstract = corpus[doc_id]
            gold = {i for r in rationales for i in r.sentences}
            ranked = _ranked_sentences(claim_tokens, abstract, gold)
            for rationale in rationales:
                decisive.append(
                    example(claim, abstract, rationale.sentences, rationale.label, "rationale")
                )
                # At inference the selector often adds a neighbouring sentence, and the
                # verdict should survive that.
                if ranked:
                    extended = [*rationale.sentences, ranked[0]]
                    decisive.append(
                        example(
                            claim, abstract, extended, rationale.label, "rationale_plus_distractor"
                        )
                    )
            for index in ranked[:negatives_per_abstract]:
                negatives.append(
                    example(claim, abstract, [index], NEI, "evidence_abstract_sentence")
                )
            if len(ranked) >= 3:
                negatives.append(example(claim, abstract, ranked[:3], NEI, "evidence_abstract_set"))

        for doc_id in claim.cited_doc_ids:
            if doc_id in claim.evidence:
                continue
            abstract = corpus[doc_id]
            ranked = _ranked_sentences(claim_tokens, abstract, set())
            for index in ranked[:negatives_per_abstract]:
                negatives.append(
                    example(claim, abstract, [index], NEI, "cited_no_evidence_sentence")
                )
            if len(ranked) >= 3:
                negatives.append(example(claim, abstract, ranked[:3], NEI, "cited_no_evidence_set"))

        cited = set(claim.cited_doc_ids)
        uncited = [d for d in retrieved.get(claim.id, ()) if d not in cited][:retrieved_abstracts]
        for doc_id in uncited:
            abstract = corpus[doc_id]
            ranked = _ranked_sentences(claim_tokens, abstract, set())
            negatives.append(example(claim, abstract, ranked[:1], NEI, "retrieved_sentence"))
            if len(ranked) >= 3:
                negatives.append(example(claim, abstract, ranked[:3], NEI, "retrieved_set"))

    rng = random.Random(seed)
    keep = min(len(negatives), round(nei_per_decisive * len(decisive)))
    examples = decisive + rng.sample(negatives, keep)
    rng.shuffle(examples)
    return examples


def build_selector_examples(
    claims: Sequence[Claim],
    corpus: Mapping[int, Abstract],
    retrieved: Mapping[int, Sequence[int]],
    *,
    retrieved_abstracts: int = 3,
    sentences_per_retrieved_abstract: int = 3,
    seed: int = 13,
) -> list[tuple[str, str, float]]:
    """(claim, sentence, is_rationale) triples, in shuffled order."""
    examples: list[tuple[str, str, float]] = []
    for claim in claims:
        for doc_id in claim.cited_doc_ids:
            gold = {i for r in claim.evidence.get(doc_id, ()) for i in r.sentences}
            for index, sentence in enumerate(corpus[doc_id].sentences):
                examples.append((claim.text, sentence, 1.0 if index in gold else 0.0))

        claim_tokens = set(tokenize(claim.text))
        cited = set(claim.cited_doc_ids)
        uncited = [d for d in retrieved.get(claim.id, ()) if d not in cited][:retrieved_abstracts]
        for doc_id in uncited:
            abstract = corpus[doc_id]
            ranked = _ranked_sentences(claim_tokens, abstract, set())
            for index in ranked[:sentences_per_retrieved_abstract]:
                examples.append((claim.text, abstract.sentences[index], 0.0))
    random.Random(seed).shuffle(examples)
    return examples


def balanced_class_weights(examples: Sequence[VerdictExample]) -> list[float]:
    """Inverse-frequency weights in LABELS order, normalised to a mean of one."""
    counts = [sum(e.label == label for e in examples) for label in LABELS]
    if 0 in counts:
        raise ValueError(f"a class has no examples: {dict(zip(LABELS, counts, strict=True))}")
    raw = [len(examples) / (len(LABELS) * c) for c in counts]
    return [w * len(LABELS) / sum(raw) for w in raw]
