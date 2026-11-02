"""The single evaluation behind every reported number.

``--split val`` is a rehearsal that exercises the whole script on tuning data and
writes to results/rehearsal_val. ``--split test --confirm-final`` reads the held-out
claims, writes results/final, and refuses to run if a manifest is already there, so a
second look at the test set takes a deliberate deletion that the git history would show.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from claimlens.calibration import (
    area_under_precision_coverage,
    bootstrap_interval,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    precision_coverage_curve,
    reliability_bins,
    softmax,
)
from claimlens.data import (
    LABELS,
    NEI,
    SUPPORTS,
    Claim,
    DataPaths,
    load_corpus,
    load_holdout,
    load_qrels,
    load_tuning_splits,
)
from claimlens.encoders import CrossEncoder
from claimlens.evaluation import (
    abstract_predictions,
    apply_policy,
    claim_logits,
    claim_report,
    confidence_and_correct,
    gold_indices,
    run_sentence_pass,
    selective_summary,
    sentence_idf,
    tfidf_top_sentences,
)
from claimlens.fusion import reciprocal_rank_fusion
from claimlens.metrics_ir import evaluate_run, micro_recall_at_k, ndcg_at_k
from claimlens.metrics_verdict import AbstractPrediction, classification_report, scifact_scores
from claimlens.models import MODELS, local_path
from claimlens.pipeline import (
    build_retriever,
    load_policy,
    load_retrieval_config,
    load_verdict_config,
    reranker_path,
)
from claimlens.retrieval import best_rerank_score
from claimlens.verdict import NLIModel, VerdictConfig, format_premise

N_RESAMPLES = 2000


def write(directory: Path, name: str, payload: object) -> None:
    (directory / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Mean per-claim difference a - b, with a bootstrap interval over claims."""
    difference = a - b
    low, high = bootstrap_interval(
        lambda idx: float(difference[idx].mean()), len(difference), n_resamples=N_RESAMPLES
    )
    return {"mean": float(difference.mean()), "ci95_low": low, "ci95_high": high}


def rerank(reranker: CrossEncoder, claims, corpus, pools) -> dict[int, list[int]]:
    run = {}
    for claim in claims:
        pool = pools[claim.id]
        scores = reranker.score(claim.text, [corpus[d].full_text for d in pool])
        run[claim.id] = [pool[i] for i in np.lexsort((np.arange(len(pool)), -scores))]
    return run


def evaluate_retrieval(claims, qrels, corpus, retriever, stock_reranker, output_hits):
    config = retriever.config
    lexical = {c.id: retriever.bm25_ranking(c.text) for c in claims}
    dense = {c.id: retriever.dense_ranking(c.text) for c in claims}
    fused = {}
    for claim in claims:
        ranked = reciprocal_rank_fusion(
            [lexical[claim.id], dense[claim.id]],
            k=config.rrf_k,
            weights=[1.0, config.dense_weight],
        )
        fused[claim.id] = [d for d, _ in ranked]
    pools = {c.id: fused[c.id][: config.rerank_depth] for c in claims}

    def with_tail(claim_id: int, head: list[int]) -> list[int]:
        # Only the top of the fused list is reranked. Keeping the rest in fused order
        # stops Recall@100 from being capped by the rerank depth.
        seen = set(head)
        return head + [d for d in fused[claim_id] if d not in seen]

    stock = rerank(stock_reranker, claims, corpus, pools)
    finetuned = rerank(retriever.reranker, claims, corpus, pools)
    runs = {
        "bm25": lexical,
        "dense": dense,
        "rrf": fused,
        "rrf+stock_cross_encoder": {c.id: with_tail(c.id, stock[c.id]) for c in claims},
        "rrf+finetuned_cross_encoder": {c.id: with_tail(c.id, finetuned[c.id]) for c in claims},
        "full": {c.id: with_tail(c.id, [h.doc_id for h in output_hits[c.id]]) for c in claims},
    }

    evidence = {c.id: list(c.evidence) for c in claims if c.evidence}
    table = {}
    for name, run in runs.items():
        table[name] = evaluate_run(run, qrels)
        for k in (3, 5):
            table[name][f"evidence_micro_recall@{k}"] = micro_recall_at_k(run, evidence, k)

    per_claim = {
        name: np.array([ndcg_at_k(run.get(c.id, ()), qrels[c.id], 10) for c in claims])
        for name, run in runs.items()
    }
    comparisons = {
        f"{a}_minus_{b}": paired_difference(per_claim[a], per_claim[b])
        for a, b in (
            ("rrf", "bm25"),
            ("rrf", "dense"),
            ("full", "bm25"),
            ("full", "dense"),
            ("full", "rrf"),
            ("full", "rrf+finetuned_cross_encoder"),
        )
    }
    return {
        "note": "Relevance comes from BEIR qrels, which list every cited abstract, including "
        "abstracts that contain no evidence for the claim. evidence_micro_recall is pooled over "
        "abstracts with annotated evidence, the definition behind VerT5erini Table 6.",
        "oracle_check_micro_recall@3": micro_recall_at_k(
            {c.id: list(c.evidence) for c in claims}, evidence, 3
        ),
        "config": config.to_dict(),
        "n_claims": len(claims),
        "runs": table,
        "paired_ndcg@10": comparisons,
    }


def macro_f1_interval(claims: list[Claim], logits: np.ndarray) -> list[float]:
    gold = [c.verdict for c in claims]
    predicted = [LABELS[i] for i in logits.argmax(axis=1)]

    def statistic(indices: np.ndarray) -> float:
        report = classification_report([gold[i] for i in indices], [predicted[i] for i in indices])
        return float(report["macro_f1"])

    return list(bootstrap_interval(statistic, len(claims), n_resamples=N_RESAMPLES))


def evaluate_system(claims, corpus, hits, selector, nli: NLIModel, config: VerdictConfig):
    """Claim-level and official scores for one model with its tuned settings."""
    documents = {c.id: [h.doc_id for h in hits[c.id][: config.top_k]] for c in claims}
    judged = run_sentence_pass(selector, claims, documents, corpus).judge(
        nli,
        threshold=config.rationale_threshold,
        max_sentences=config.max_rationale_sentences,
        nei_weight=config.nli_nei_weight,
    )
    logits = claim_logits(claims, judged, top_k=config.top_k, nei_bias=config.nei_bias)
    predictions = abstract_predictions(claims, judged, logits, top_k=config.top_k)
    return (
        judged,
        logits,
        {
            "claim_level": claim_report(claims, logits),
            "claim_macro_f1_ci95": macro_f1_interval(claims, logits),
            "official_scifact": scifact_scores(claims, predictions),
        },
    )


def oracle_rationale_labels(claims, corpus, nli: NLIModel):
    """Label prediction with gold rationales, the setup of VerT5erini Table 8.

    Evidence abstracts get their annotated sentences. Abstracts cited by a claim with
    no evidence have none, so the two sentences closest to the claim by TF-IDF stand in.
    """
    idf = sentence_idf(corpus)
    pairs, gold = [], []
    for claim in claims:
        if claim.evidence:
            for doc_id, rationales in claim.evidence.items():
                indices = sorted({i for r in rationales for i in r.sentences})
                abstract = corpus[doc_id]
                chosen = [abstract.sentences[i] for i in indices]
                pairs.append((format_premise(abstract.title, chosen), claim.text))
                gold.append(rationales[0].label)
        else:
            for doc_id in claim.cited_doc_ids:
                abstract = corpus[doc_id]
                indices = sorted(tfidf_top_sentences(claim.text, abstract, idf, 2))
                chosen = [abstract.sentences[i] for i in indices]
                pairs.append((format_premise(abstract.title, chosen), claim.text))
                gold.append(NEI)
    predicted = [LABELS[i] for i in nli.logits(pairs).argmax(axis=1)]
    return classification_report(gold, predicted)


def oracle_abstract_scores(claims, corpus, selector, nli: NLIModel, config: VerdictConfig):
    """Gold evidence abstracts are given, and the system picks rationale and label."""
    documents = {c.id: list(c.evidence) for c in claims if c.evidence}
    subset = [c for c in claims if c.evidence]
    judged = run_sentence_pass(selector, subset, documents, corpus).judge(
        nli,
        threshold=config.rationale_threshold,
        max_sentences=config.max_rationale_sentences,
        nei_weight=config.nli_nei_weight,
    )
    predictions = {
        claim_id: {
            j.doc_id: AbstractPrediction(j.label, j.rationale) for j in items if j.is_decisive
        }
        for claim_id, items in judged.items()
    }
    return scifact_scores(subset, predictions)


def evaluate_calibration(claims, logits, temperature, policy):
    gold = gold_indices(claims)
    raw, correct = confidence_and_correct(claims, logits)
    scaled, _ = confidence_and_correct(claims, logits, temperature=temperature)
    estimated = policy.confidence(softmax(logits / temperature))

    def summary(confidences):
        def ece(indices):
            return expected_calibration_error(confidences[indices], correct[indices])

        return {
            "ece_10_bins": expected_calibration_error(confidences, correct),
            "ece_ci95": list(bootstrap_interval(ece, len(claims), n_resamples=N_RESAMPLES)),
            "ece_15_bins": expected_calibration_error(confidences, correct, n_bins=15),
            "mean_confidence": float(confidences.mean()),
            "accuracy": float(correct.mean()),
            "bins": reliability_bins(confidences, correct),
        }

    def proper_scores(scaled_logits):
        return {
            "nll": negative_log_likelihood(scaled_logits, gold),
            "brier": brier_score(softmax(scaled_logits), gold),
        }

    return {
        "n_claims": len(claims),
        "fitted_on": "calib",
        "temperature": temperature,
        "uncalibrated": summary(raw) | proper_scores(logits),
        "temperature_scaled": summary(scaled) | proper_scores(logits / temperature),
        "confidence_model": summary(estimated),
    }


def evaluate_abstention(claims, judged, hits, logits, config, policy):
    scaled, correct = confidence_and_correct(claims, logits, temperature=config.temperature)
    confidences = policy.confidence(softmax(logits / config.temperature))
    decisions = apply_policy(
        claims,
        judged,
        hits,
        logits,
        top_k=config.top_k,
        temperature=config.temperature,
        policy=policy,
    )
    answered = np.array([not d.abstained for d in decisions])
    reasons: dict[str, int] = {}
    for decision in decisions:
        for abstention in decision.abstentions:
            reasons[abstention.reason.value] = reasons.get(abstention.reason.value, 0) + 1

    def accuracy_on(indices: np.ndarray) -> float:
        mask = answered[indices]
        return float(correct[indices][mask].mean()) if mask.any() else 0.0

    def coverage_on(indices: np.ndarray) -> float:
        return float(answered[indices].mean())

    retrieval_scores = np.array([best_rerank_score(hits[c.id][: config.top_k]) for c in claims])
    curve = precision_coverage_curve(confidences, correct)
    temperature_only = precision_coverage_curve(scaled, correct)
    naive = precision_coverage_curve(retrieval_scores, correct)
    answered_gold = [c.verdict for c, a in zip(claims, answered, strict=True) if a]
    answered_guess = [d.best_guess for d, a in zip(decisions, answered, strict=True) if a]
    return {
        "n_claims": len(claims),
        "policy": policy.to_dict(),
        "answer_everything": {
            "coverage": 1.0,
            "accuracy": float(correct.mean()),
        },
        "operating_point": selective_summary(claims, decisions),
        "operating_point_ci95": {
            "selective_accuracy": list(
                bootstrap_interval(accuracy_on, len(claims), n_resamples=N_RESAMPLES)
            ),
            "coverage": list(bootstrap_interval(coverage_on, len(claims), n_resamples=N_RESAMPLES)),
        },
        "operating_point_report": classification_report(answered_gold, answered_guess),
        "abstention_reasons": reasons,
        "curve": curve,
        "area_under_curve": area_under_precision_coverage(curve),
        "ranked_by_temperature_scaled_probability_curve": temperature_only,
        "ranked_by_temperature_scaled_probability_area": area_under_precision_coverage(
            temperature_only
        ),
        "ranked_by_retrieval_score_curve": naive,
        "ranked_by_retrieval_score_area": area_under_precision_coverage(naive),
    }


def evaluate_withheld_evidence(claims, corpus, hits, selector, nli, config, policy):
    """Delete every cited abstract and count the decisive answers that survive.

    With the cited abstracts gone, these claims have no annotated evidence left, so
    each SUPPORTS or REFUTES is an answer the data cannot back up. SciFact abstracts
    may still bear on a claim without being cited, so this is an upper bound on the
    harm, not an exact count.
    """
    subset = [c for c in claims if c.evidence]
    reduced = {c.id: [h for h in hits[c.id] if h.doc_id not in c.cited_doc_ids] for c in subset}
    documents = {c.id: [h.doc_id for h in reduced[c.id][: config.top_k]] for c in subset}
    judged = run_sentence_pass(selector, subset, documents, corpus).judge(
        nli,
        threshold=config.rationale_threshold,
        max_sentences=config.max_rationale_sentences,
        nei_weight=config.nli_nei_weight,
    )
    logits = claim_logits(subset, judged, top_k=config.top_k, nei_bias=config.nei_bias)
    decisions = apply_policy(
        subset,
        judged,
        reduced,
        logits,
        top_k=config.top_k,
        temperature=config.temperature,
        policy=policy,
    )
    without_policy = sum(d.best_guess != NEI for d in decisions)
    with_policy = sum(d.verdict is not None and d.verdict != NEI for d in decisions)
    reasons: dict[str, int] = {}
    for decision in decisions:
        for abstention in decision.abstentions:
            reasons[abstention.reason.value] = reasons.get(abstention.reason.value, 0) + 1
    return {
        "n_claims": len(subset),
        "decisive_without_abstention": without_policy,
        "decisive_rate_without_abstention": without_policy / len(subset),
        "decisive_with_abstention": with_policy,
        "decisive_rate_with_abstention": with_policy / len(subset),
        "answered_not_enough_info": sum(d.verdict == NEI for d in decisions),
        "abstained": sum(d.abstained for d in decisions),
        "abstention_reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--confirm-final", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path("runs/nli/best"))
    parser.add_argument("--selector", type=Path, default=Path("runs/selector/best"))
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.split == "test":
        output = args.results / "final"
        if (output / "manifest.json").exists():
            raise SystemExit(f"{output} already holds a final run; the test set is used once")
        claims, qrels = load_holdout(args.data, confirm_final_evaluation=args.confirm_final)
    else:
        output = args.results / "rehearsal_val"
        claims = load_tuning_splits(args.data)["val"]
        train_qrels = load_qrels(DataPaths(args.data).qrels_train)
        qrels = {c.id: train_qrels[c.id] for c in claims}
    output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    corpus = load_corpus(DataPaths(args.data).corpus)
    retrieval_config = load_retrieval_config(args.configs)
    verdict_config = load_verdict_config(args.configs)
    stock_config = VerdictConfig(**json.loads((args.configs / "verdict_stock.json").read_text()))
    policy = load_policy(args.configs)

    retriever = build_retriever(corpus, retrieval_config)
    stock_reranker = CrossEncoder.load(local_path("ms-marco-minilm"))
    depth = 20
    hits = {c.id: retriever.search(c.text, depth) for c in claims}

    retrieval = evaluate_retrieval(claims, qrels, corpus, retriever, stock_reranker, hits)
    write(output, "retrieval.json", retrieval)
    print(json.dumps(retrieval["runs"]["full"], indent=2))

    nli = NLIModel.load(args.model)
    selector = CrossEncoder.load(args.selector, max_length=128)
    judged, logits, finetuned = evaluate_system(claims, corpus, hits, selector, nli, verdict_config)

    majority = {label: sum(c.verdict == label for c in claims) for label in LABELS}
    majority_label = max(LABELS, key=lambda label: (majority[label], label == SUPPORTS))
    verdict = {
        "n_claims": len(claims),
        "gold_distribution": majority,
        "note": "Claim-level macro-F1 covers SUPPORTS, REFUTES and NOT_ENOUGH_INFO with every "
        "claim answered. official_scifact uses the leaderboard evaluator's definitions.",
        "finetuned": finetuned,
        "majority_baseline": classification_report(
            [c.verdict for c in claims], [majority_label] * len(claims)
        ),
        "oracle_rationale_labels": oracle_rationale_labels(claims, corpus, nli),
        "oracle_abstract_scores": oracle_abstract_scores(
            claims, corpus, selector, nli, verdict_config
        ),
    }

    stock_nli = NLIModel.load(local_path("deberta-nli"))
    _, _, stock_scores = evaluate_system(
        claims, corpus, hits, stock_reranker, stock_nli, stock_config
    )
    verdict["stock_weights"] = stock_scores
    verdict["stock_nli_oracle_rationale_labels"] = oracle_rationale_labels(
        claims, corpus, stock_nli
    )
    del stock_nli
    write(output, "verdict.json", verdict)
    print(json.dumps(verdict["finetuned"]["official_scifact"], indent=2))

    write(
        output,
        "calibration.json",
        evaluate_calibration(claims, logits, verdict_config.temperature, policy),
    )
    write(
        output,
        "precision_coverage.json",
        evaluate_abstention(claims, judged, hits, logits, verdict_config, policy),
    )
    write(
        output,
        "withheld_evidence.json",
        evaluate_withheld_evidence(claims, corpus, hits, selector, nli, verdict_config, policy),
    )

    gold = gold_indices(claims)
    probabilities = softmax(logits / verdict_config.temperature)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for claim, row, label in zip(claims, probabilities, gold, strict=True):
            record = {
                "id": claim.id,
                "gold": LABELS[label],
                "probabilities": dict(zip(LABELS, row.round(6).tolist(), strict=True)),
                "documents": [j.doc_id for j in judged.get(claim.id, [])[: verdict_config.top_k]],
            }
            handle.write(json.dumps(record) + "\n")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    chip = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    write(
        output,
        "manifest.json",
        {
            "split": args.split,
            "n_claims": len(claims),
            "description": "SciFact dev claims, which are also the BEIR SciFact test queries"
            if args.split == "test"
            else "Validation claims carved from SciFact train (rehearsal only)",
            "finished_unix_time": time.time(),
            "seconds": time.time() - started,
            "git_commit": commit,
            "machine": chip or platform.platform(),
            "nli_weights_sha256": sha256(args.model / "model.safetensors"),
            "selector_weights_sha256": sha256(args.selector / "model.safetensors"),
            "reranker_weights_sha256": sha256(
                reranker_path(retrieval_config.reranker) / "model.safetensors"
            ),
            "base_models": {key: MODELS[key].repo for key in ("deberta-nli", "ms-marco-minilm")},
            "configs": {
                "retrieval": retrieval_config.to_dict(),
                "verdict": verdict_config.to_dict(),
                "stock_verdict": stock_config.to_dict(),
                "abstention": policy.to_dict(),
            },
        },
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
