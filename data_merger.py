"""
data_merger.py
--------------
Tải về và merge 2 loại dataset:
  1. LLM-as-a-Judge data (điểm do LLM chấm)
  2. Human Evaluation data (điểm do 3 người đánh giá)

Quy trình merge:
  - Tải 3 bộ Human Evaluation (evaluator_1, 2, 3), concat lại
  - Tính điểm trung bình (mean) của 3 người → human_avg_<criterion>
  - Merge với LLM-judge data theo conversation_id
  - Lưu dataset hợp nhất ra CSV

Output columns (ngoài các cột text gốc):
  llm_<criterion>        : điểm LLM judge
  human_avg_<criterion>  : điểm trung bình 3 human
  human_<E>_<criterion>  : điểm từng evaluator (E = 1,2,3)
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(col: str) -> str:
    """'Continuity and Context Understanding' → 'Continuity_and_Context_Understanding'."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_")


def _rename_score_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Đổi tên 4 cột điểm bằng prefix chỉ định."""
    rename_map = {col: f"{prefix}_{_slugify(col)}" for col in config.SCORE_COLS}
    return df.rename(columns=rename_map)


# ── Load datasets ─────────────────────────────────────────────────────────────

def load_llm_judge_dataset() -> pd.DataFrame:
    """Tải LLM-as-a-Judge dataset từ HuggingFace."""
    log.info("Đang tải LLM-Judge dataset: %s", config.HF_LLM_JUDGE_DATASET)
    ds = load_dataset(config.HF_LLM_JUDGE_DATASET, split="train")
    df = ds.to_pandas()
    log.info("  → %d dòng, columns: %s", len(df), list(df.columns))
    return df


def load_human_eval_datasets() -> pd.DataFrame:
    """
    Tải 3 Human Evaluation datasets, đánh nhãn evaluator, concat.

    Returns
    -------
    DataFrame với cột 'evaluator_id' (1/2/3) và các cột điểm gốc.
    """
    frames = []
    for i, hf_path in enumerate(config.HF_HUMAN_EVAL_DATASETS, start=1):
        log.info("Đang tải Human Eval dataset %d: %s", i, hf_path)
        ds = load_dataset(hf_path, split="train")
        df = ds.to_pandas()
        df["evaluator_id"] = i
        log.info("  → %d dòng", len(df))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Tổng human eval: %d dòng (3 evaluators)", len(combined))
    return combined


# ── Merge ─────────────────────────────────────────────────────────────────────

def build_merged_dataset(
    llm_df: pd.DataFrame,
    human_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge LLM-judge và Human Evaluation datasets theo conversation_id.

    Chiến lược:
      - Human: pivot từng evaluator thành cột riêng + tính avg
      - LLM: rename cột điểm thành prefix 'llm_'
      - Merge inner join theo conversation_id

    Returns
    -------
    DataFrame hợp nhất.
    """
    log.info("=== Bắt đầu merge datasets ===")

    # 1. Chuẩn bị LLM dataframe
    llm = llm_df.copy()
    # Chỉ giữ 1 hàng mỗi conversation_id nếu bị trùng
    llm = llm.drop_duplicates(subset=["conversation_id"])
    llm_text_cols = [
        "conversation_id", "conversation_stage", "instruction",
        "history", "history_summary", "client_question",
        "ground_truth", "generated_answer",
    ]
    # Lấy các cột text + điểm
    available_llm_cols = [c for c in llm_text_cols if c in llm.columns]
    llm_scores = [c for c in config.SCORE_COLS if c in llm.columns]
    llm = llm[available_llm_cols + llm_scores].copy()
    llm = _rename_score_cols(llm, prefix="llm")

    # 2. Chuẩn bị Human dataframe – pivot từng evaluator
    human = human_df.copy()

    # Lấy ra các cột cần thiết
    human_base_cols = ["conversation_id", "evaluator_id"]
    human_score_cols = [c for c in config.SCORE_COLS if c in human.columns]

    human_slim = human[human_base_cols + human_score_cols].copy()

    # Pivot: 1 hàng mỗi conversation_id, cột = evaluator × criterion
    pivot_frames = []
    for eid, group in human_slim.groupby("evaluator_id"):
        g = group.drop(columns=["evaluator_id"]).drop_duplicates(
            subset=["conversation_id"]
        )
        g = g.rename(
            columns={
                col: f"human_e{eid}_{_slugify(col)}"
                for col in human_score_cols
            }
        )
        pivot_frames.append(g)

    human_wide = pivot_frames[0]
    for pf in pivot_frames[1:]:
        human_wide = human_wide.merge(pf, on="conversation_id", how="outer")

    # Tính điểm trung bình 3 evaluators
    for col in human_score_cols:
        slug = _slugify(col)
        eval_cols = [f"human_e{i}_{slug}" for i in range(1, 4)
                     if f"human_e{i}_{slug}" in human_wide.columns]
        human_wide[f"human_avg_{slug}"] = human_wide[eval_cols].mean(axis=1)

    log.info(
        "Human wide table: %d hàng, %d cột",
        len(human_wide), len(human_wide.columns),
    )

    # 3. Merge
    merged = llm.merge(human_wide, on="conversation_id", how="inner")
    log.info(
        "Sau inner merge: %d hàng, %d cột",
        len(merged), len(merged.columns),
    )

    # 4. Thống kê missing
    total_cells = len(merged) * len(config.SCORE_COLS)
    llm_missing = sum(
        merged[f"llm_{_slugify(c)}"].isna().sum()
        for c in config.SCORE_COLS
        if f"llm_{_slugify(c)}" in merged.columns
    )
    log.info(
        "LLM missing scores: %d / %d (%.1f%%)",
        llm_missing, total_cells, 100 * llm_missing / max(total_cells, 1),
    )

    return merged


# ── Save & load ────────────────────────────────────────────────────────────────

def save_merged(df: pd.DataFrame, path: str = config.MERGED_DATASET_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("Đã lưu merged dataset → %s (%d dòng)", path, len(df))


def load_merged(path: str = config.MERGED_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    log.info("Đã tải merged dataset ← %s (%d dòng)", path, len(df))
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_merge(force: bool = False) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline merge. Nếu file đã tồn tại thì load lại
    (trừ khi force=True).
    """
    if not force and os.path.exists(config.MERGED_DATASET_PATH):
        log.info("File merged đã tồn tại, đang load lại (dùng force=True để tạo mới).")
        return load_merged()

    llm_df   = load_llm_judge_dataset()
    human_df = load_human_eval_datasets()
    merged   = build_merged_dataset(llm_df, human_df)
    save_merged(merged)
    return merged


if __name__ == "__main__":
    df = run_merge(force=True)
    print("\n--- Thông tin dataset sau merge ---")
    print(df.info())
    print("\nMột vài dòng mẫu:")
    print(df.head(3).to_string())
