"""
main.py
-------
Entry point chạy toàn bộ pipeline LLM-as-a-Judge:

  1. data_merger.py    → Merge & lưu merged_dataset.csv
  2. baseline_evaluator.py → Đánh giá baseline (điểm sẵn có + API zero-shot)
  3. improved_judge.py → Chạy improved judge (few-shot + CoT + rubric)
  4. So sánh & lưu metrics_comparison.csv

Vì giới hạn 20 RPD (20 requests/ngày), mỗi pipeline chạy MAX_SAMPLES_PER_RUN mẫu.

Chạy:
  python main.py [--force-merge] [--skip-baseline-api] [--skip-improved]
                 [--self-consistency] [--samples N]
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

import config
from data_merger import run_merge
from baseline_evaluator import evaluate_existing_llm_scores, run_baseline_api_evaluation
from improved_judge import run_improved_evaluation
from metrics_utils import compute_all_metrics, compute_metrics_per_criterion, _slugify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_comparison_table(
    merged_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    improved_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Tạo bảng so sánh metrics giữa:
      A. LLM scores sẵn có trong dataset (baseline_existing)
      B. Baseline API (zero-shot)
      C. Improved Judge (few-shot + CoT + rubric)

    Returns
    -------
    DataFrame với rows = tiêu chí + overall, columns = metric × judge_type
    """
    rows = []

    def add_rows(label: str, human_cols: list, judge_cols: list, df: pd.DataFrame):
        for hc, jc in zip(human_cols, judge_cols):
            sub = df[[hc, jc]].dropna()
            if len(sub) == 0:
                continue
            m = compute_all_metrics(sub[hc].values, sub[jc].values)
            criterion = hc.replace("human_avg_", "").replace("_", " ").title()
            rows.append({"Judge": label, "Criterion": criterion, **m})

        # Overall
        all_t, all_p = [], []
        for hc, jc in zip(human_cols, judge_cols):
            sub = df[[hc, jc]].dropna()
            all_t.extend(sub[hc].values)
            all_p.extend(sub[jc].values)
        if all_t:
            m = compute_all_metrics(np.array(all_t), np.array(all_p))
            rows.append({"Judge": label, "Criterion": "OVERALL", **m})

    # A: LLM dataset gốc (tất cả dữ liệu, không giới hạn n_samples)
    hcols_all = [f"human_avg_{_slugify(c)}" for c in config.SCORE_COLS
                 if f"human_avg_{_slugify(c)}" in merged_df.columns]
    jcols_all = [f"llm_{_slugify(c)}" for c in config.SCORE_COLS
                 if f"llm_{_slugify(c)}" in merged_df.columns]
    if hcols_all and jcols_all:
        add_rows("Baseline_Existing_LLM", hcols_all, jcols_all, merged_df)

    # B: Baseline API
    if baseline_df is not None:
        hcols = [f"human_avg_{_slugify(c)}" for c in config.SCORE_COLS
                 if f"human_avg_{_slugify(c)}" in baseline_df.columns]
        jcols = [f"baseline_api_{_slugify(c)}" for c in config.SCORE_COLS
                 if f"baseline_api_{_slugify(c)}" in baseline_df.columns]
        if hcols and jcols:
            add_rows("Baseline_API_ZeroShot", hcols, jcols, baseline_df)

    # C: Improved
    if improved_df is not None:
        hcols = [f"human_avg_{_slugify(c)}" for c in config.SCORE_COLS
                 if f"human_avg_{_slugify(c)}" in improved_df.columns]
        jcols = [f"improved_{_slugify(c)}" for c in config.SCORE_COLS
                 if f"improved_{_slugify(c)}" in improved_df.columns]
        if hcols and jcols:
            add_rows("Improved_FewShot_CoT_Rubric", hcols, jcols, improved_df)

    if not rows:
        log.warning("Không có dữ liệu để so sánh!")
        return pd.DataFrame()

    comparison = pd.DataFrame(rows)
    return comparison


def print_summary(comparison: pd.DataFrame) -> None:
    """In bảng tóm tắt so sánh đẹp ra console."""
    if comparison.empty:
        return

    print("\n" + "=" * 100)
    print(" BẢNG SO SÁNH METRICS: BASELINE vs IMPROVED JUDGE ".center(100, "="))
    print("=" * 100)

    # Chỉ in OVERALL row
    overall = comparison[comparison["Criterion"] == "OVERALL"]
    if not overall.empty:
        print("\n📊 OVERALL METRICS (macro-average over all criteria):\n")
        display_cols = [
            "Judge", "pearson", "spearman", "kendall_tau",
            "mae", "rmse", "exact_match", "within_1_acc",
            "weighted_kappa", "bias",
        ]
        display_cols = [c for c in display_cols if c in overall.columns]
        print(overall[display_cols].to_string(index=False, float_format="%.4f"))

    print("\n" + "-" * 100)
    print("\n📋 PER-CRITERION METRICS:\n")

    for judge in comparison["Judge"].unique():
        sub = comparison[comparison["Judge"] == judge]
        sub_crit = sub[sub["Criterion"] != "OVERALL"]
        if sub_crit.empty:
            continue
        print(f"\n  ── {judge} ──")
        display_cols = ["Criterion", "pearson", "spearman", "mae", "within_1_acc", "weighted_kappa"]
        display_cols = [c for c in display_cols if c in sub_crit.columns]
        print(sub_crit[display_cols].to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-a-Judge Pipeline")
    p.add_argument("--force-merge", action="store_true",
                   help="Tải lại và merge dataset dù đã có file")
    p.add_argument("--skip-baseline-api", action="store_true",
                   help="Bỏ qua bước gọi API baseline (tiết kiệm RPD)")
    p.add_argument("--skip-improved", action="store_true",
                   help="Bỏ qua bước chạy improved judge")
    p.add_argument("--self-consistency", action="store_true",
                   help="Dùng self-consistency (gọi API 2 lần/sample, tốn 2x RPD)")
    p.add_argument("--samples", type=int, default=config.MAX_SAMPLES_PER_RUN,
                   help=f"Số mẫu gọi API (default: {config.MAX_SAMPLES_PER_RUN})")
    return p.parse_args()


def main():
    args = parse_args()

    Path(config.DATA_DIR).mkdir(exist_ok=True)
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)

    log.info("=" * 70)
    log.info("  LLM-as-a-Judge Pipeline".center(70))
    log.info("=" * 70)

    # ── Step 1: Merge datasets ─────────────────────────────────────────────────
    log.info("\n[STEP 1] Merge Datasets")
    merged_df = run_merge(force=args.force_merge)
    log.info("Merged dataset: %d dòng, %d cột", len(merged_df), len(merged_df.columns))

    # ── Step 2: Baseline - Đánh giá LLM scores có sẵn ─────────────────────────
    log.info("\n[STEP 2] Baseline Evaluation (LLM scores từ dataset gốc)")
    existing_metrics = evaluate_existing_llm_scores(merged_df)

    # ── Step 3: Baseline API (zero-shot) ──────────────────────────────────────
    baseline_df = None
    if not args.skip_baseline_api:
        log.info(
            "\n[STEP 3] Baseline API Evaluation (zero-shot, %d mẫu)",
            args.samples,
        )
        baseline_df = run_baseline_api_evaluation(
            merged_df,
            n_samples=args.samples,
            save_path=config.BASELINE_RESULTS_PATH,
        )
    else:
        log.info("[STEP 3] Skipped baseline API (--skip-baseline-api)")
        if os.path.exists(config.BASELINE_RESULTS_PATH):
            baseline_df = pd.read_csv(config.BASELINE_RESULTS_PATH)
            log.info("Loaded existing baseline results: %d dòng", len(baseline_df))

    # ── Step 4: Improved Judge ─────────────────────────────────────────────────
    improved_df = None
    if not args.skip_improved:
        log.info(
            "\n[STEP 4] Improved Judge Evaluation (few-shot + CoT + rubric, %d mẫu)",
            args.samples,
        )
        improved_df = run_improved_evaluation(
            merged_df,
            n_samples=args.samples,
            save_path=config.IMPROVED_RESULTS_PATH,
            use_self_consistency=args.self_consistency,
        )
    else:
        log.info("[STEP 4] Skipped improved judge (--skip-improved)")
        if os.path.exists(config.IMPROVED_RESULTS_PATH):
            improved_df = pd.read_csv(config.IMPROVED_RESULTS_PATH)
            log.info("Loaded existing improved results: %d dòng", len(improved_df))

    # ── Step 5: Comparison Table ───────────────────────────────────────────────
    log.info("\n[STEP 5] Building Comparison Table")
    comparison = build_comparison_table(merged_df, baseline_df, improved_df)

    if not comparison.empty:
        comparison.to_csv(config.METRICS_COMPARISON_PATH, index=False, encoding="utf-8-sig")
        log.info("Đã lưu comparison → %s", config.METRICS_COMPARISON_PATH)
        print_summary(comparison)

    log.info("\n✅ Pipeline hoàn thành!")
    log.info("  - Merged dataset  : %s", config.MERGED_DATASET_PATH)
    log.info("  - Baseline results: %s", config.BASELINE_RESULTS_PATH)
    log.info("  - Improved results: %s", config.IMPROVED_RESULTS_PATH)
    log.info("  - Metrics compare : %s", config.METRICS_COMPARISON_PATH)


if __name__ == "__main__":
    main()
