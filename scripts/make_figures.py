"""Draw the README figures from a results directory. Every mark comes from those files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e4dd"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

RUN_LABELS = {
    "bm25": "BM25 (hand-written)",
    "dense": "Dense (bge-base)",
    "rrf": "BM25 + dense, RRF",
    "rrf+stock_cross_encoder": "RRF, stock reranker",
    "rrf+finetuned_cross_encoder": "RRF, fine-tuned reranker",
    "full": "RRF, fine-tuned reranker, late fusion",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "serif",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
        }
    )


def precision_coverage(results: Path, output: Path) -> None:
    data = json.loads((results / "precision_coverage.json").read_text())
    curve = data["curve"]
    naive = data["ranked_by_retrieval_score_curve"]
    point = data["operating_point"]

    figure, axes = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    axes.grid(True, axis="y")
    axes.plot(
        [p["coverage"] for p in curve],
        [p["precision"] for p in curve],
        color=SERIES_1,
        linewidth=2,
        solid_capstyle="round",
        label="Ranked by estimated P(correct)",
    )
    axes.plot(
        [p["coverage"] for p in naive],
        [p["precision"] for p in naive],
        color=SERIES_2,
        linewidth=2,
        solid_capstyle="round",
        label="Ranked by reranker score",
    )
    axes.axhline(
        data["answer_everything"]["accuracy"],
        color=MUTED,
        linewidth=1,
        linestyle=(0, (4, 3)),
        label="Answers every claim",
    )
    axes.plot(
        [point["coverage"]],
        [point["selective_accuracy"]],
        marker="o",
        markersize=8,
        markerfacecolor=SERIES_1,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        linestyle="none",
        label="Deployed policy, thresholds from calib",
    )
    axes.annotate(
        f"{point['coverage']:.0%} answered, {point['selective_accuracy']:.0%} correct",
        (point["coverage"], point["selective_accuracy"]),
        xytext=(8, 10),
        textcoords="offset points",
        color=INK,
        fontsize=9,
    )
    axes.set_xlim(0, 1.02)
    axes.set_ylim(0.3, 1.02)
    axes.set_xlabel("Coverage (fraction of claims answered)")
    axes.set_ylabel("Accuracy on answered claims")
    axes.set_title(f"Selective accuracy on {data['n_claims']} held-out claims")
    axes.legend(frameon=False, loc="lower left", fontsize=8, labelcolor=MUTED)
    figure.tight_layout()
    figure.savefig(output / "precision_coverage.png")
    plt.close(figure)


def reliability(results: Path, output: Path) -> None:
    data = json.loads((results / "calibration.json").read_text())
    figure, panels = plt.subplots(1, 2, figsize=(6.4, 3.3), dpi=200, sharex=True, sharey=True)
    for axes, key, title, color in (
        (panels[0], "uncalibrated", "Before", SERIES_2),
        (panels[1], "confidence_model", "After calibration", SERIES_1),
    ):
        block = data[key]
        axes.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        for item in block["bins"]:
            axes.scatter(
                item["confidence"],
                item["accuracy"],
                s=12 + item["count"] * 1.5,
                color=color,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=3,
            )
        axes.grid(True)
        axes.set_xlim(0.2, 1.02)
        axes.set_ylim(0, 1.05)
        axes.set_aspect("equal", adjustable="box")
        axes.set_title(f"{title}: ECE {block['ece_10_bins']:.3f}", fontsize=10)
        axes.set_xlabel("Mean confidence in bin")
    panels[0].set_ylabel("Accuracy in bin")
    figure.tight_layout()
    figure.savefig(output / "reliability.png")
    plt.close(figure)


def retrieval(results: Path, output: Path) -> None:
    data = json.loads((results / "retrieval.json").read_text())["runs"]
    names = [n for n in RUN_LABELS if n in data][::-1]
    values = [data[n]["ndcg@10"] for n in names]
    figure, axes = plt.subplots(figsize=(6.4, 2.6), dpi=200)
    axes.barh(
        [RUN_LABELS[n] for n in names],
        values,
        color=SERIES_1,
        height=0.55,
        edgecolor=SURFACE,
        linewidth=2,
    )
    for index, value in enumerate(values):
        axes.text(value + 0.008, index, f"{value:.3f}", va="center", color=INK, fontsize=9)
    axes.set_xlim(0, 1)
    axes.grid(True, axis="x")
    axes.set_axisbelow(True)
    axes.set_xlabel("nDCG@10")
    axes.set_title("Retrieval on the BEIR SciFact test queries")
    axes.tick_params(axis="y", length=0)
    figure.tight_layout()
    figure.savefig(output / "retrieval.png")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/final"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    style()
    precision_coverage(args.results, args.output)
    reliability(args.results, args.output)
    retrieval(args.results, args.output)


if __name__ == "__main__":
    main()
