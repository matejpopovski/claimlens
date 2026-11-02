"""Loading SciFact and carving the splits.

The split layout is the one fact a reader of the results must keep in mind:

* BEIR's test queries are exactly the 300 SciFact dev claims, and BEIR's corpus is
  the SciFact corpus, so one held-out set serves retrieval and verdict evaluation.
* SciFact's own test labels are unreleased, so that dev split is the final test set
  and every tuning decision comes from a validation split carved out of train.
* The ``cross_validation`` folds shipped with SciFact mix train and dev claims and
  are never read here, because using them would leak the held-out claims.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SUPPORTS = "SUPPORTS"
REFUTES = "REFUTES"
NEI = "NOT_ENOUGH_INFO"
LABELS = (SUPPORTS, REFUTES, NEI)

_RAW_LABELS = {"SUPPORT": SUPPORTS, "CONTRADICT": REFUTES}

Split = Literal["fit", "val", "calib"]
SPLIT_FRACTIONS: Mapping[str, float] = {"fit": 0.70, "val": 0.15, "calib": 0.15}
SPLIT_SEED = 13


@dataclass(frozen=True)
class Abstract:
    doc_id: int
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.text}"


@dataclass(frozen=True)
class Rationale:
    sentences: tuple[int, ...]
    label: str


@dataclass(frozen=True)
class Claim:
    id: int
    text: str
    cited_doc_ids: tuple[int, ...] = ()
    evidence: Mapping[int, tuple[Rationale, ...]] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        """Claim-level gold label: SciFact has no claim whose abstracts disagree."""
        labels = {r.label for rationales in self.evidence.values() for r in rationales}
        if not labels:
            return NEI
        if len(labels) > 1:
            raise ValueError(f"claim {self.id} has conflicting gold labels")
        return labels.pop()


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def parse_claim(row: Mapping) -> Claim:
    evidence = {
        int(doc_id): tuple(
            Rationale(tuple(sorted(e["sentences"])), _RAW_LABELS[e["label"]]) for e in entries
        )
        for doc_id, entries in row.get("evidence", {}).items()
    }
    return Claim(
        id=int(row["id"]),
        text=row["claim"],
        cited_doc_ids=tuple(int(d) for d in row.get("cited_doc_ids", ())),
        evidence=evidence,
    )


def load_corpus(path: Path) -> dict[int, Abstract]:
    corpus = {}
    for row in _read_jsonl(path):
        doc_id = int(row["doc_id"])
        sentences = tuple(s.strip() for s in row["abstract"])
        corpus[doc_id] = Abstract(doc_id, row["title"].strip(), sentences)
    return corpus


def load_claims(path: Path) -> list[Claim]:
    return [parse_claim(row) for row in _read_jsonl(path)]


def load_qrels(path: Path) -> dict[int, dict[int, int]]:
    """Read a BEIR qrels TSV. These list cited abstracts, including evidence-free ones."""
    qrels: dict[int, dict[int, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            query_id, doc_id, grade = line.split()
            qrels[int(query_id)][int(doc_id)] = int(grade)
    return dict(qrels)


def _claim_groups(claims: Sequence[Claim]) -> list[list[Claim]]:
    """Connected components of claims that share a cited abstract.

    SciFact negations reuse the source claim's abstract, so a random split would put
    the same abstract, and near-duplicate claims, on both sides.
    """
    parent = {claim.id: claim.id for claim in claims}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_doc: dict[int, int] = {}
    for claim in claims:
        for doc_id in claim.cited_doc_ids:
            if doc_id in by_doc:
                parent[find(claim.id)] = find(by_doc[doc_id])
            else:
                by_doc[doc_id] = claim.id

    groups: dict[int, list[Claim]] = defaultdict(list)
    for claim in claims:
        groups[find(claim.id)].append(claim)
    return [sorted(g, key=lambda c: c.id) for _, g in sorted(groups.items())]


def split_train_claims(
    claims: Sequence[Claim],
    *,
    fractions: Mapping[str, float] = SPLIT_FRACTIONS,
    seed: int = SPLIT_SEED,
) -> dict[str, list[Claim]]:
    """Split claims with no shared cited abstract, roughly stratified by verdict.

    Groups are placed greedily, each going to the split furthest below its target
    count for that group's most common verdict.
    """
    groups = _claim_groups(claims)
    random.Random(seed).shuffle(groups)

    totals = Counter(claim.verdict for claim in claims)
    targets = {
        name: {label: frac * totals[label] for label in totals} for name, frac in fractions.items()
    }
    placed: dict[str, list[Claim]] = {name: [] for name in fractions}
    filled = {name: Counter() for name in fractions}
    for group in groups:
        verdict_counts = Counter(claim.verdict for claim in group)
        label = verdict_counts.most_common(1)[0][0]
        name = max(fractions, key=lambda n: targets[n][label] - filled[n][label])
        placed[name].extend(group)
        filled[name].update(verdict_counts)
    return {name: sorted(items, key=lambda c: c.id) for name, items in placed.items()}


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def corpus(self) -> Path:
        return self.root / "raw/scifact/data/corpus.jsonl"

    @property
    def claims_train(self) -> Path:
        return self.root / "raw/scifact/data/claims_train.jsonl"

    @property
    def claims_dev(self) -> Path:
        return self.root / "raw/scifact/data/claims_dev.jsonl"

    @property
    def qrels_train(self) -> Path:
        return self.root / "raw/beir/scifact/qrels/train.tsv"

    @property
    def qrels_test(self) -> Path:
        return self.root / "raw/beir/scifact/qrels/test.tsv"


def load_tuning_splits(root: Path | str = "data") -> dict[str, list[Claim]]:
    """The only claims that tuning code may read: fit, val and calib, all from train."""
    return split_train_claims(load_claims(DataPaths(Path(root)).claims_train))


def load_holdout(root: Path | str = "data", *, confirm_final_evaluation: bool = False):
    """Return the held-out claims and BEIR test qrels.

    The flag makes reading the test set a visible act at the call site. Only
    scripts/final_eval.py passes it.
    """
    if not confirm_final_evaluation:
        raise PermissionError("the held-out set is reserved for scripts/final_eval.py")
    paths = DataPaths(Path(root))
    return load_claims(paths.claims_dev), load_qrels(paths.qrels_test)
