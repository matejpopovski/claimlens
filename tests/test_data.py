import json

import pytest

from claimlens.data import (
    NEI,
    REFUTES,
    SUPPORTS,
    Claim,
    Rationale,
    load_claims,
    load_corpus,
    load_holdout,
    load_qrels,
    parse_claim,
    split_train_claims,
)


def synthetic_claims(n=400):
    """Claims in pairs that share a cited abstract, like SciFact's negations."""
    claims = []
    for i in range(n):
        doc = i // 2
        if i % 3 == 0:
            evidence = {doc: (Rationale((0,), SUPPORTS),)}
        elif i % 3 == 1:
            evidence = {doc: (Rationale((1,), REFUTES),)}
        else:
            evidence = {}
        claims.append(Claim(i, f"claim {i}", (doc,), evidence))
    return claims


def test_parse_claim_maps_labels_and_integer_keys():
    claim = parse_claim(
        {
            "id": 7,
            "claim": "x",
            "cited_doc_ids": [3],
            "evidence": {"3": [{"sentences": [2, 1], "label": "CONTRADICT"}]},
        }
    )
    assert claim.evidence == {3: (Rationale((1, 2), REFUTES),)}
    assert claim.verdict == REFUTES


def test_claim_without_evidence_is_not_enough_info():
    assert Claim(1, "x", (2,), {}).verdict == NEI


def test_conflicting_gold_labels_are_refused():
    claim = Claim(
        1,
        "x",
        (2, 3),
        {2: (Rationale((0,), SUPPORTS),), 3: (Rationale((0,), REFUTES),)},
    )
    with pytest.raises(ValueError, match="conflicting"):
        _ = claim.verdict


def test_split_is_a_partition():
    claims = synthetic_claims()
    splits = split_train_claims(claims)
    ids = [c.id for part in splits.values() for c in part]
    assert sorted(ids) == sorted(c.id for c in claims)


def test_no_cited_abstract_crosses_a_split_boundary():
    splits = split_train_claims(synthetic_claims())
    docs = {name: {d for c in part for d in c.cited_doc_ids} for name, part in splits.items()}
    assert not docs["fit"] & docs["val"]
    assert not docs["fit"] & docs["calib"]
    assert not docs["val"] & docs["calib"]


def test_split_grouping_is_transitive():
    # 0-1 share doc 10 and 1-2 share doc 11, so all three must stay together.
    claims = [
        Claim(0, "a", (10,)),
        Claim(1, "b", (10, 11)),
        Claim(2, "c", (11,)),
        *[Claim(i, str(i), (100 + i,)) for i in range(3, 40)],
    ]
    splits = split_train_claims(claims)
    homes = [name for name, part in splits.items() if any(c.id in (0, 1, 2) for c in part)]
    assert len(homes) == 1


def test_split_is_deterministic_and_respects_fractions():
    claims = synthetic_claims()
    first = split_train_claims(claims)
    second = split_train_claims(claims)
    assert {k: [c.id for c in v] for k, v in first.items()} == {
        k: [c.id for c in v] for k, v in second.items()
    }
    assert len(first["fit"]) == pytest.approx(0.70 * len(claims), abs=0.05 * len(claims))
    assert len(first["val"]) == pytest.approx(0.15 * len(claims), abs=0.05 * len(claims))


def test_holdout_requires_an_explicit_flag(tmp_path):
    with pytest.raises(PermissionError):
        load_holdout(tmp_path)


def test_jsonl_and_qrels_readers(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"doc_id": 5, "title": " T ", "abstract": [" one ", "two"], "structured": False})
        + "\n"
    )
    assert load_corpus(corpus)[5].sentences == ("one", "two")
    assert load_corpus(corpus)[5].full_text == "T one two"

    claims = tmp_path / "claims.jsonl"
    claims.write_text(json.dumps({"id": 1, "claim": "c", "evidence": {}, "cited_doc_ids": [5]}))
    assert load_claims(claims)[0].cited_doc_ids == (5,)

    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n1\t5\t1\n1\t6\t1\n")
    assert load_qrels(qrels) == {1: {5: 1, 6: 1}}
