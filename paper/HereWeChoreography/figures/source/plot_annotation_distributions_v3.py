"""Render compact, paired-data CVPR statistics from the frozen aggregate audit.

Run: python figures/source/plot_annotation_distributions_v3.py
Requires numpy and matplotlib only; no raw data, network, or model inference.
Histogram bins are categorical frequency bars, not equal-width density bins.
The final audit edge is an overflow sentinel and is never plotted numerically.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path(__file__).resolve().parent
OUT = SOURCE.parent
AUDIT = SOURCE / "dataset_distribution_audit.json"
FIGURE_SIZE = (6.875, 1.80)
TEAL, NAVY, PURPLE = "#238B8E", "#26445C", "#8063AA"
INK, MUTED, GRID = "#263746", "#61717C", "#E9EDF0"


def validate(data):
    assert data["source_revision"] == "1ca58b4ebc7f9457fb7dc7db587c05f3e79b767a"
    assert data["schema"] == "dataset_distribution_audit.v1"
    assert all(v["matches_committed_audit"] is True for v in data["source_files"].values())
    assert {k: v["n"] for k, v in data["source_files"].items()} == {
        "train": 198315, "val": 11098, "test": 11254}
    paired = data["paired_val_test"]["all"]
    n = paired["n"]
    assert n == paired["final"]["n"] == data["pairing"]["strict_paired_unique_ids"] == 22352
    assert data["pairing"]["all_ids_used_not_quality_selected"] is True
    for name in ("original_duplicate_id_records", "final_duplicate_id_records",
                 "missing_final_unique_ids", "missing_original_unique_ids", "source_or_split_mismatch"):
        assert data["pairing"][name] == 0, name
    histograms = (paired["original_event_histogram"], paired["final"]["english_body_word_histogram"])
    for histogram in histograms:
        assert histogram["n"] == n == sum(histogram["counts"])
        assert histogram["interval"] == "[left,right)"
        assert len(histogram["edges"]) == len(histogram["counts"]) + 1
        assert all(isinstance(v, int) and v >= 0 for v in histogram["counts"])
    assert histograms[0]["edges"] == [0, 10, 20, 40, 60, 80, 100000]
    assert histograms[1]["edges"] == [0, 60, 100, 140, 180, 100000]
    assert set(paired["original_bodypart_event_counts"]) == {"wrist", "elbow", "knee"}
    total = paired["original_event_total"]
    assert sum(paired["original_bodypart_event_counts"].values()) == total
    assert sum(paired["original_event_statement_counts"].values()) == total
    assert np.isclose(paired["original_event_statements_per_clip"]["mean"] * n, total)
    assert paired["original_style_counts"] == paired["final"]["style_counts"]
    assert paired["original_event_histogram"]["counts"] == [108, 543, 6504, 10507, 4198, 492]
    assert paired["final"]["english_body_word_histogram"]["counts"] == [407, 3442, 8761, 6903, 2839]
    assert paired["original_bodypart_event_counts"] == {"wrist": 636663, "elbow": 312964, "knee": 105511}
    assert np.isclose(paired["final"]["english_body_words"]["mean"], 135.327979599141,
                      atol=1e-12, rtol=0)
    assert round(total / n, 2) == 47.21
    assert [round(v / total * 100, 1) for v in paired["original_bodypart_event_counts"].values()] == [60.3, 29.7, 10.0]
    return paired


def style_axes(ax):
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#B7C2CA")
        ax.spines[edge].set_linewidth(0.5)
    ax.tick_params(colors=MUTED, length=2.0, width=0.5, pad=2)


def histogram(ax, title, labels, counts, denominator, color, mean, unit):
    x = np.arange(len(counts))
    shares = np.asarray(counts, dtype=float) / denominator * 100
    ax.bar(x, shares, width=0.74, color=color, edgecolor="none", zorder=3)
    for xi, share in zip(x, shares):
        ax.text(xi, share + 1.15, f"{share:.1f}", ha="center", va="bottom",
                fontsize=7.5, color=INK)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 54), yticks=[0, 25, 50],
           xlim=(-0.6, len(x) - 0.4), xlabel=unit, ylabel="Clips (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_title(title, loc="left", pad=23)
    ax.text(0, 1.04, f"mean {mean:.2f}", transform=ax.transAxes,
            color=MUTED, fontsize=8.2, ha="left", va="bottom")
    style_axes(ax)


def main():
    assert hashlib.sha256(AUDIT.read_bytes()).hexdigest() == "bef81f195f1fc88372ee6b3a8206818d043e28721fff97f9ab4a077254637052", "Frozen audit changed"
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    p = validate(data)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE,
                             gridspec_kw={"width_ratios": [1.14, 1.34, 0.82]})
    fig.subplots_adjust(left=0.06, right=0.987, bottom=0.24, top=0.72, wspace=0.37)

    histogram(axes[0], "(i) Source event count",
              ["0–9", "10–19", "20–39", "40–59", "60–79", "≥80"],
              p["original_event_histogram"]["counts"], p["n"], TEAL,
              p["original_event_statements_per_clip"]["mean"], "Rule-event statements")
    histogram(axes[1], "(ii) Refined caption length",
              ["0–59", "60–99", "100–139", "140–179", "≥180"],
              p["final"]["english_body_word_histogram"]["counts"], p["n"], PURPLE,
              p["final"]["english_body_words"]["mean"], "English body words")

    ax = axes[2]
    labels = ["Wrist", "Elbow", "Knee"]
    counts = np.array([p["original_bodypart_event_counts"][s.lower()] for s in labels])
    shares = counts / p["original_event_total"] * 100
    y = np.arange(3)
    ax.barh(y, shares, height=0.49, color=[NAVY, TEAL, PURPLE], edgecolor="none", zorder=3)
    for yi, value in zip(y, shares):
        ax.text(value + 2.0, yi, f"{value:.1f}%", ha="left", va="center",
                fontsize=9, fontweight="bold", color=INK)
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 84), xticks=[0, 30, 60],
           xlabel="Source events (%)", ylim=(2.5, -0.5))
    ax.set_title("(iii) Body-part shares", loc="left", pad=23)
    ax.text(0, 1.04, "event occurrences", transform=ax.transAxes,
            color=MUTED, fontsize=8.2, ha="left", va="bottom")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    style_axes(ax)
    ax.tick_params(axis="y", length=0, labelsize=8.5, pad=4)
    ax.spines["left"].set_visible(False)

    for extension in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"annotation_distributions_v3.{extension}", dpi=360)
    plt.close(fig)
    # Print source-bound metadata for the manuscript caption without another data artifact.
    summary = {
        "source_revision": data["source_revision"],
        "audit_sha256": hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
        "paired_clips": p["n"], "quality_filter_applied": False,
        "source_event_total": p["original_event_total"],
        "event_counts": p["original_event_histogram"]["counts"],
        "word_counts": p["final"]["english_body_word_histogram"]["counts"],
        "mean_events": p["original_event_statements_per_clip"]["mean"],
        "mean_body_words": p["final"]["english_body_words"]["mean"],
        "bodypart_counts": dict(zip(labels, map(int, counts))),
        "bodypart_percentages": dict(zip(labels, map(float, shares))),
        "canvas_inches": FIGURE_SIZE,
        "body_words_exclude_style_header": True,
        "overflow_bins": {"events": ">=80", "words": ">=180"},
        "interpretation": "Descriptive frequencies only; not action diversity or correctness gains.",
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
