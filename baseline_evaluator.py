"""
baseline_evaluator.py
---------------------
Đánh giá LLM-as-a-Judge BASELINE:
  - Dùng điểm LLM-judge sẵn có trong dataset (không gọi API mới)
    để tính tất cả metrics so với human evaluation.
  - Ngoài ra, chọn ngẫu nhiên MAX_SAMPLES_PER_RUN mẫu và gọi API Gemini
    với prompt đơn giản (zero-shot, không có few-shot, không có CoT)
    để tạo "điểm baseline mới" – phân biệt với điểm LLM sẵn có trong dataset.

Output:
  - results/baseline_results.csv  : điểm baseline từng mẫu
  - In ra console bảng tổng hợp metrics
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types
import pandas as pd

import config
from metrics_utils import (
    compute_all_metrics,
    compute_metrics_per_criterion,
    _slugify,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Cấu hình Gemini ───────────────────────────────────────────────────────────
_CLIENT = genai.Client(api_key=config.GEMINI_API_KEY)

# ── Prompt baseline (zero-shot, không few-shot, không CoT) ────────────────────
BASELINE_SYSTEM_PROMPT = """You are an expert evaluator for customer service AI responses.
Evaluate the given AI-generated response based on 4 criteria.
Return ONLY a valid JSON object with numeric scores (1-5) for each criterion.
No explanations, no markdown, just the JSON object."""

BASELINE_USER_TEMPLATE = """Context:
Instruction: {instruction}
Conversation History: {history}
Client Question: {client_question}
Ground Truth Answer: {ground_truth}
AI Generated Answer: {generated_answer}

Score the AI Generated Answer on a scale of 1 (very poor) to 5 (excellent) for:
1. Human-Likeness: How natural and human-like is the response?
2. Continuity and Context Understanding: Does it maintain conversation flow?
3. Tone and Clarity: Is it clear, professional, and appropriately toned?
4. Task Appropriateness: Does it address the customer's question effectively?

Return ONLY this JSON (no markdown, no extra text):
{{
  "Human-Likeness": <score>,
  "Continuity and Context Understanding": <score>,
  "Tone and Clarity": <score>,
  "Task Appropriateness": <score>
}}"""


def call_baseline_judge(row: pd.Series) -> dict[str, float] | None:
    """
    Gọi Gemini API với prompt baseline đơn giản (zero-shot).
    Trả về dict điểm hoặc None nếu lỗi.
    """
    prompt = BASELINE_USER_TEMPLATE.format(
        instruction=str(row.get("instruction", ""))[:800],
        history=str(row.get("history", ""))[:600],
        client_question=str(row.get("client_question", ""))[:400],
        ground_truth=str(row.get("ground_truth", ""))[:400],
        generated_answer=str(row.get("generated_answer", ""))[:600],
    )

    try:
        response = _CLIENT.models.generate_content(
            model=config.JUDGE_MODEL,
            contents=[BASELINE_SYSTEM_PROMPT + "\n\n" + prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
            ),
        )
        raw = response.text.strip()

        # Trích xuất JSON từ response
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not json_match:
            log.warning("Không tìm thấy JSON trong response: %s", raw[:200])
            return None

        scores = json.loads(json_match.group())

        # Validate và normalize
        result = {}
        for col in config.SCORE_COLS:
            val = scores.get(col)
            if val is None:
                log.warning("Thiếu key '%s' trong response", col)
                return None
            val = float(val)
            if not 1 <= val <= 5:
                log.warning("Score out of range [1,5]: %s=%s", col, val)
                val = max(1.0, min(5.0, val))
            result[col] = val
        return result

    except json.JSONDecodeError as e:
        log.error("JSON parse error: %s | raw: %s", e, raw[:200])

        return None
    except Exception as e:
        log.error("API call error: %s", e)
        return None


# ── Đánh giá baseline trên dataset sẵn có (không gọi API) ────────────────────

def evaluate_existing_llm_scores(merged_df: pd.DataFrame) -> dict:
    """
    Đánh giá điểm LLM đã có sẵn trong dataset (không gọi API mới).
    So sánh llm_<criterion> với human_avg_<criterion>.

    Returns
    -------
    dict: {'per_criterion': DataFrame, 'overall': dict}
    """
    log.info("=== Đánh giá điểm LLM baseline (từ dataset gốc) ===")

    human_cols = [f"human_avg_{_slugify(c)}" for c in config.SCORE_COLS]
    judge_cols = [f"llm_{_slugify(c)}" for c in config.SCORE_COLS]

    # Lọc cột tồn tại
    valid = [(h, j) for h, j in zip(human_cols, judge_cols)
             if h in merged_df.columns and j in merged_df.columns]

    if not valid:
        log.error("Không tìm thấy cột phù hợp! Kiểm tra lại tên cột trong merged_df.")
        return {}

    human_cols_valid = [v[0] for v in valid]
    judge_cols_valid = [v[1] for v in valid]

    per_criterion = compute_metrics_per_criterion(
        merged_df, human_cols_valid, judge_cols_valid
    )

    # Overall (gộp tất cả tiêu chí)
    all_true = []
    all_pred = []
    for hc, jc in zip(human_cols_valid, judge_cols_valid):
        sub = merged_df[[hc, jc]].dropna()
        all_true.extend(sub[hc].values)
        all_pred.extend(sub[jc].values)

    import numpy as np
    overall = compute_all_metrics(
        y_true=pd.array(all_true, dtype=float),
        y_pred=pd.array(all_pred, dtype=float),
        prefix="overall",
    )

    log.info("\n--- Metrics per criterion (Baseline - dataset gốc) ---")
    log.info("\n%s", per_criterion.to_string())
    log.info("\n--- Overall Metrics ---")
    for k, v in overall.items():
        log.info("  %-30s : %.4f", k, v)

    return {"per_criterion": per_criterion, "overall": overall}


# ── Gọi API tạo baseline scores mới (giới hạn số lượng) ──────────────────────

def run_baseline_api_evaluation(
    merged_df: pd.DataFrame,
    n_samples: int = config.MAX_SAMPLES_PER_RUN,
    save_path: str = config.BASELINE_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Gọi Gemini API với prompt baseline (zero-shot) trên n_samples mẫu.
    So sánh kết quả với điểm human.

    Returns
    -------
    DataFrame kết quả với cột baseline_<criterion>
    """
    log.info("=== Chạy Baseline API Evaluation (%d mẫu) ===", n_samples)

    # Lấy mẫu ngẫu nhiên có seed cố định để tái lập được
    sample = merged_df.sample(
        n=min(n_samples, len(merged_df)),
        random_state=42,
    ).reset_index(drop=True)

    results = []
    for idx, row in sample.iterrows():
        log.info(
            "[%d/%d] Đang xử lý conversation_id: %s",
            idx + 1, len(sample), row.get("conversation_id", "?"),
        )

        scores = call_baseline_judge(row)

        record = {
            "conversation_id": row.get("conversation_id"),
            "conversation_stage": row.get("conversation_stage"),
        }

        # Human avg scores
        for col in config.SCORE_COLS:
            slug = _slugify(col)
            record[f"human_avg_{slug}"] = row.get(f"human_avg_{slug}")

        # LLM scores từ dataset gốc
        for col in config.SCORE_COLS:
            slug = _slugify(col)
            record[f"llm_{slug}"] = row.get(f"llm_{slug}")

        # Baseline API scores
        if scores:
            for col in config.SCORE_COLS:
                slug = _slugify(col)
                record[f"baseline_api_{slug}"] = scores.get(col)
        else:
            for col in config.SCORE_COLS:
                slug = _slugify(col)
                record[f"baseline_api_{slug}"] = None

        results.append(record)

        # Rate limit: sleep để tránh vượt 5 RPM
        if idx < len(sample) - 1:
            log.info("  → Sleep %ds (rate limit 5 RPM)", config.RATE_LIMIT_SLEEP)
            time.sleep(config.RATE_LIMIT_SLEEP)

    result_df = pd.DataFrame(results)

    # Tính metrics baseline API
    human_cols = [f"human_avg_{_slugify(c)}" for c in config.SCORE_COLS]
    baseline_cols = [f"baseline_api_{_slugify(c)}" for c in config.SCORE_COLS]

    valid = [
        (h, b) for h, b in zip(human_cols, baseline_cols)
        if h in result_df.columns and b in result_df.columns
    ]

    if valid:
        log.info("\n--- Metrics per criterion (Baseline API - zero-shot) ---")
        per_criterion = compute_metrics_per_criterion(
            result_df,
            [v[0] for v in valid],
            [v[1] for v in valid],
        )
        log.info("\n%s", per_criterion.to_string())

    # Lưu kết quả
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    log.info("Đã lưu baseline results → %s", save_path)

    return result_df


if __name__ == "__main__":
    from data_merger import load_merged
    merged = load_merged()

    # 1. Đánh giá điểm LLM gốc trong dataset (không tốn API call)
    existing_metrics = evaluate_existing_llm_scores(merged)

    # 2. Gọi API baseline (giới hạn số lượng vì 20 RPD)
    baseline_df = run_baseline_api_evaluation(merged)
    print("\nBaseline API results (mẫu đầu):")
    print(baseline_df.head(3).to_string())
