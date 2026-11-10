"""
visualize_results.py
--------------------
Tạo biểu đồ so sánh Baseline vs Improved Judge dựa trên
file metrics_comparison.csv đã được tạo bởi main.py.

Biểu đồ được lưu vào thư mục results/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FIGURES_DIR = Path(config.RESULTS_DIR) / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Màu cho từng judge type
COLORS = {
    "Baseline_Existing_LLM":       "#4878CF",   # xanh
    "Baseline_API_ZeroShot":       "#D65F5F",   # đỏ
    "Improved_FewShot_CoT_Rubric": "#6ACC65",   # xanh lá
}

LABEL_MAP = {
    "Baseline_Existing_LLM":       "Baseline (Dataset LLM)",
    "Baseline_API_ZeroShot":       "Baseline (API Zero-Shot)",
    "Improved_FewShot_CoT_Rubric": "Improved (Few-Shot + CoT + Rubric)",
}


def plot_overall_metrics(df: pd.DataFrame) -> None:
    """Bar chart so sánh overall metrics giữa các judge."""
    overall = df[df["Criterion"] == "OVERALL"].copy()
    if overall.empty:
        log.warning("Không có OVERALL row!")
        return

    metrics_to_plot = ["pearson", "spearman", "kendall_tau",
                       "exact_match", "within_1_acc", "weighted_kappa"]
    metrics_to_plot = [m for m in metrics_to_plot if m in overall.columns]

    judges = overall["Judge"].tolist()
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (_, row) in enumerate(overall.iterrows()):
        judge = row["Judge"]
        vals = [row[m] for m in metrics_to_plot]
        offset = (i - len(judges) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=LABEL_MAP.get(judge, judge),
            color=COLORS.get(judge, f"C{i}"),
            alpha=0.85,
            edgecolor="white",
        )
        # Thêm giá trị lên bar
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5, rotation=45,
            )

    ax.set_title("Overall Metrics: Baseline vs Improved LLM-as-a-Judge", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metrics_to_plot],
        fontsize=10,
    )
    ax.set_ylabel("Score (↑ better for correlation/match; ↓ for mae/rmse)", fontsize=10)
    ax.set_ylim(-0.1, 1.15)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / "overall_metrics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Đã lưu → %s", out)
    plt.close()


def plot_per_criterion_heatmap(df: pd.DataFrame, metric: str = "pearson") -> None:
    """Heatmap: rows = tiêu chí, cols = judge type."""
    crit_df = df[df["Criterion"] != "OVERALL"].copy()
    if crit_df.empty or metric not in crit_df.columns:
        return

    pivot = crit_df.pivot(index="Criterion", columns="Judge", values=metric)
    pivot.columns = [LABEL_MAP.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label=metric.capitalize())

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(f"Per-Criterion {metric.capitalize()} Correlation", fontsize=13, fontweight="bold")

    # Thêm giá trị vào ô
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="black" if 0.3 < val < 0.8 else "white")

    plt.tight_layout()
    out = FIGURES_DIR / f"per_criterion_{metric}_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Đã lưu → %s", out)
    plt.close()


def plot_error_metrics(df: pd.DataFrame) -> None:
    """So sánh MAE và RMSE cho overall."""
    overall = df[df["Criterion"] == "OVERALL"].copy()
    if overall.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["mae", "rmse"]):
        judges = overall["Judge"].tolist()
        vals = [overall.loc[overall["Judge"] == j, metric].values[0] for j in judges]
        colors = [COLORS.get(j, "gray") for j in judges]
        labels = [LABEL_MAP.get(j, j) for j in judges]

        bars = ax.bar(range(len(judges)), vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(judges)))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_title(f"Overall {metric.upper()} (↓ better)", fontsize=12)
        ax.set_ylabel(metric.upper())
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=10,
            )

    plt.suptitle("Error Metrics: Baseline vs Improved Judge", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "error_metrics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Đã lưu → %s", out)
    plt.close()


def plot_score_distribution(baseline_df: pd.DataFrame | None,
                            improved_df: pd.DataFrame | None) -> None:
    """Histogram phân phối điểm (human vs judge) cho từng tiêu chí."""
    if baseline_df is None and improved_df is None:
        return

    criteria = config.SCORE_COLS
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, criterion in zip(axes, criteria):
        slug = config._slugify(criterion) if hasattr(config, "_slugify") else \
               criterion.replace(" ", "_").replace("-", "_")
        # Import từ metrics_utils
        from metrics_utils import _slugify
        slug = _slugify(criterion)

        hcol = f"human_avg_{slug}"
        bcol = f"baseline_api_{slug}"
        icol = f"improved_{slug}"

        ax.set_title(criterion, fontsize=10, fontweight="bold")
        bins = np.arange(0.5, 6, 0.5)

        # Human distribution (từ baseline_df nếu có)
        ref_df = baseline_df if baseline_df is not None else improved_df
        if ref_df is not None and hcol in ref_df.columns:
            h_data = ref_df[hcol].dropna()
            ax.hist(h_data, bins=bins, alpha=0.5, label="Human Avg", color="gray", density=True)

        if baseline_df is not None and bcol in baseline_df.columns:
            b_data = baseline_df[bcol].dropna()
            ax.hist(b_data, bins=bins, alpha=0.6, label="Baseline (zero-shot)",
                    color=COLORS["Baseline_API_ZeroShot"], density=True)

        if improved_df is not None and icol in improved_df.columns:
            i_data = improved_df[icol].dropna()
            ax.hist(i_data, bins=bins, alpha=0.6, label="Improved",
                    color=COLORS["Improved_FewShot_CoT_Rubric"], density=True)

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Score Distribution: Human vs Judge", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "score_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Đã lưu → %s", out)
    plt.close()


def run_all_visualizations() -> None:
    """Chạy tất cả visualizations."""
    # Load comparison
    if not Path(config.METRICS_COMPARISON_PATH).exists():
        log.error("Chưa có metrics_comparison.csv! Hãy chạy main.py trước.")
        return

    comp_df = pd.read_csv(config.METRICS_COMPARISON_PATH)
    log.info("Loaded comparison: %d rows", len(comp_df))

    baseline_df = None
    improved_df = None

    if Path(config.BASELINE_RESULTS_PATH).exists():
        baseline_df = pd.read_csv(config.BASELINE_RESULTS_PATH)
        log.info("Loaded baseline: %d rows", len(baseline_df))

    if Path(config.IMPROVED_RESULTS_PATH).exists():
        improved_df = pd.read_csv(config.IMPROVED_RESULTS_PATH)
        log.info("Loaded improved: %d rows", len(improved_df))

    plot_overall_metrics(comp_df)
    plot_per_criterion_heatmap(comp_df, metric="pearson")
    plot_per_criterion_heatmap(comp_df, metric="spearman")
    plot_error_metrics(comp_df)
    plot_score_distribution(baseline_df, improved_df)

    log.info("\n✅ Tất cả biểu đồ đã lưu vào: %s", FIGURES_DIR)


if __name__ == "__main__":
    run_all_visualizations()
