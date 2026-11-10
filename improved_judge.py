"""
improved_judge.py
-----------------
Xây dựng LLM-as-a-Judge CẢI TIẾN với các kỹ thuật:

  1. Few-Shot Prompting    – cung cấp ví dụ có điểm để anchor thang điểm
  2. Chain-of-Thought (CoT) – yêu cầu LLM giải thích rồi mới cho điểm
  3. Rubric-based Scoring  – cung cấp rubric chi tiết từng mức điểm 1-5
  4. Self-Consistency      – gọi API 2 lần, lấy trung bình (tiết kiệm RPD)
  5. Detailed Criteria     – định nghĩa rõ ràng từng tiêu chí

Kỳ vọng: điểm tương quan cao hơn với human evaluation so với baseline.

Lưu kết quả vào results/improved_results.csv.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from google import genai
from google.genai import types as genai_types
import numpy as np
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

# ── Rubric chi tiết từng tiêu chí ────────────────────────────────────────────
RUBRIC = """
=== SCORING RUBRIC ===

[1] Human-Likeness (1-5)
  5: Completely natural, indistinguishable from a human agent. Warm, conversational, empathetic.
  4: Mostly natural with minor robotic phrasing.
  3: Mixed—some human-like moments but noticeably AI-generated at times.
  2: Mostly formulaic, stiff, or overly formal.
  1: Clearly machine-generated; cold, repetitive, or awkward.

[2] Continuity and Context Understanding (1-5)
  5: Perfectly references prior context; seamlessly continues the conversation; no gaps.
  4: Good contextual awareness with minor oversights.
  3: Partially acknowledges context; occasional non-sequiturs.
  2: Poor context use; ignores important prior information.
  1: Completely ignores conversation history; treats inquiry in isolation.

[3] Tone and Clarity (1-5)
  5: Crystal-clear, professional yet friendly, perfectly calibrated tone.
  4: Clear and mostly appropriate; minor tone mismatches.
  3: Generally understandable but occasionally ambiguous or off-tone.
  2: Often unclear or inappropriate tone (too casual or too formal).
  1: Confusing, contradictory, or highly inappropriate tone.

[4] Task Appropriateness (1-5)
  5: Directly and completely addresses the customer's specific question with accurate information.
  4: Addresses the main question; misses minor details.
  3: Partially addresses the question; some aspects ignored.
  2: Tangentially related; fails to solve the core request.
  1: Completely off-topic or provides incorrect information.
"""

# ── Few-shot examples (anchor điểm) ──────────────────────────────────────────
FEW_SHOT_EXAMPLES = """
=== FEW-SHOT EXAMPLES ===

--- Example 1 (Low quality response) ---
Client Question: "My order hasn't arrived yet, it's been 2 weeks."
Generated Answer: "Thank you for contacting us. We will look into it."
Reasoning:
- Human-Likeness: 2 — Generic, no empathy, clearly scripted.
- Continuity: 2 — Doesn't reference any order details or history.
- Tone and Clarity: 3 — Clear but cold and unhelpful.
- Task Appropriateness: 1 — Does not address the delay at all; no next steps.
Scores: {{"Human-Likeness": 2, "Continuity and Context Understanding": 2, "Tone and Clarity": 3, "Task Appropriateness": 1}}

--- Example 2 (High quality response) ---
Client Question: "I ordered the blue shirt (order #4521), when will it arrive?"
Generated Answer: "I'm so sorry for the wait on order #4521! I've checked and it appears there was a brief delay at our warehouse. Your blue shirt is now on its way and should arrive by Thursday. I'll send you a tracking link right now. Is there anything else I can help you with?"
Reasoning:
- Human-Likeness: 5 — Warm, empathetic, natural conversation flow.
- Continuity: 5 — References specific order number and item from customer's message.
- Tone and Clarity: 5 — Professional, friendly, and crystal clear.
- Task Appropriateness: 5 — Directly answers the question with specific date and next steps.
Scores: {{"Human-Likeness": 5, "Continuity and Context Understanding": 5, "Tone and Clarity": 5, "Task Appropriateness": 5}}

--- Example 3 (Medium quality response) ---
Client Question: "Can I change my delivery address?"
Generated Answer: "Yes, you can change your delivery address. Please go to your account settings or contact customer support within 24 hours of placing your order."
Reasoning:
- Human-Likeness: 3 — Functional but slightly robotic.
- Continuity: 3 — Answers the question but doesn't personalize.
- Tone and Clarity: 4 — Clear instructions, appropriate tone.
- Task Appropriateness: 4 — Addresses the question but could be more specific about their situation.
Scores: {{"Human-Likeness": 3, "Continuity and Context Understanding": 3, "Tone and Clarity": 4, "Task Appropriateness": 4}}
"""

# ── System prompt cải tiến ───────────────────────────────────────────────────
IMPROVED_SYSTEM_PROMPT = f"""You are an expert customer service quality evaluator with 10+ years of experience.
Your task is to evaluate AI-generated customer service responses.

{RUBRIC}

{FEW_SHOT_EXAMPLES}

IMPORTANT INSTRUCTIONS:
1. Read the rubric carefully before scoring.
2. Use Chain-of-Thought: briefly reason about each criterion (1-2 sentences).
3. Be consistent with the few-shot examples above.
4. Avoid score clustering — distinguish between 2, 3, and 4 meaningfully.
5. After reasoning, output ONLY the JSON block with scores.
"""

IMPROVED_USER_TEMPLATE = """=== CONVERSATION TO EVALUATE ===

System Instruction: {instruction}

Conversation History:
{history}

Client's Latest Question: {client_question}

Reference (Ground Truth) Answer: {ground_truth}

AI Generated Answer to Evaluate: {generated_answer}

=== YOUR EVALUATION ===
Think step-by-step about each criterion based on the rubric, then provide your scores.
Your final output MUST end with this exact JSON format (no markdown fences):
{{
  "Human-Likeness": <integer 1-5>,
  "Continuity and Context Understanding": <integer 1-5>,
  "Tone and Clarity": <integer 1-5>,
  "Task Appropriateness": <integer 1-5>
}}"""


def call_improved_judge(row: pd.Series, attempt: int = 1) -> dict[str, float] | None:
    """
    Gọi improved judge với few-shot + CoT + rubric.

    Parameters
    ----------
    row     : một dòng từ merged DataFrame
    attempt : số thứ tự lần gọi (dùng để log)
    """
    prompt = IMPROVED_USER_TEMPLATE.format(
        instruction=str(row.get("instruction", ""))[:800],
        history=str(row.get("history", ""))[:800],
        client_question=str(row.get("client_question", ""))[:500],
        ground_truth=str(row.get("ground_truth", ""))[:500],
        generated_answer=str(row.get("generated_answer", ""))[:700],
    )

    try:
        response = _CLIENT.models.generate_content(
            model=config.JUDGE_MODEL,
            contents=[IMPROVED_SYSTEM_PROMPT + "\n\n" + prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.1,   # Thấp để ổn định, không quá 0
                max_output_tokens=512,
            ),
        )
        raw = response.text.strip()
        log.debug("  [attempt %d] Raw response: %s", attempt, raw[:300])


        # Trích xuất JSON cuối cùng trong response (sau CoT)
        # Tìm JSON block cuối cùng
        json_matches = list(re.finditer(r"\{[^{}]*\}", raw, re.DOTALL))
        if not json_matches:
            log.warning("  Không tìm thấy JSON trong response (attempt %d)", attempt)
            return None

        # Lấy JSON phù hợp nhất (chứa các key cần thiết)
        scores = None
        for m in reversed(json_matches):
            try:
                candidate = json.loads(m.group())
                if all(col in candidate for col in config.SCORE_COLS):
                    scores = candidate
                    break
            except json.JSONDecodeError:
                continue

        if scores is None:
            # Thử parse toàn bộ text nếu không tìm được
            try:
                full_match = re.search(r"\{[\s\S]*\}", raw)
                if full_match:
                    scores = json.loads(full_match.group())
            except Exception:
                log.warning("  Fallback JSON parse cũng thất bại (attempt %d)", attempt)
                return None

        # Validate
        result = {}
        for col in config.SCORE_COLS:
            val = scores.get(col)
            if val is None:
                log.warning("  Thiếu key '%s' (attempt %d)", col, attempt)
                return None
            val = float(val)
            val = max(1.0, min(5.0, val))
            result[col] = val

        return result

    except Exception as e:
        log.error("  API error (attempt %d): %s", attempt, e)
        return None


def run_improved_evaluation(
    merged_df: pd.DataFrame,
    n_samples: int = config.MAX_SAMPLES_PER_RUN,
    save_path: str = config.IMPROVED_RESULTS_PATH,
    use_self_consistency: bool = False,
) -> pd.DataFrame:
    """
    Chạy improved judge trên n_samples mẫu.

    Parameters
    ----------
    merged_df           : DataFrame đã merge
    n_samples           : số mẫu gọi API (giới hạn bởi RPD)
    save_path           : đường dẫn lưu kết quả
    use_self_consistency: nếu True, gọi 2 lần và lấy trung bình
                          (tốn 2x API calls, chỉ dùng khi RPD còn đủ)
    """
    log.info(
        "=== Chạy Improved Judge Evaluation (%d mẫu, self_consistency=%s) ===",
        n_samples, use_self_consistency,
    )

    sample = merged_df.sample(
        n=min(n_samples, len(merged_df)),
        random_state=42,  # Cùng seed với baseline để so sánh cùng mẫu
    ).reset_index(drop=True)

    results = []
    total_calls = 0

    for idx, row in sample.iterrows():
        log.info(
            "[%d/%d] conversation_id: %s",
            idx + 1, len(sample), row.get("conversation_id", "?"),
        )

        # Lần gọi đầu tiên
        scores1 = call_improved_judge(row, attempt=1)
        total_calls += 1

        final_scores = scores1

        # Self-consistency: gọi lần 2, lấy trung bình nếu use_self_consistency
        if use_self_consistency and scores1 is not None:
            if idx < len(sample) - 1 or True:  # Luôn sleep giữa các call
                log.info("  Sleep %ds trước lần 2...", config.RATE_LIMIT_SLEEP)
                time.sleep(config.RATE_LIMIT_SLEEP)

            scores2 = call_improved_judge(row, attempt=2)
            total_calls += 1

            if scores2 is not None:
                # Average của 2 lần gọi
                final_scores = {
                    col: (scores1[col] + scores2[col]) / 2
                    for col in config.SCORE_COLS
                }
                log.info("  Self-consistency: avg scores %s", final_scores)

        # Build record
        record = {
            "conversation_id": row.get("conversation_id"),
            "conversation_stage": row.get("conversation_stage"),
        }

        # Human avg
        for col in config.SCORE_COLS:
            slug = _slugify(col)
            record[f"human_avg_{slug}"] = row.get(f"human_avg_{slug}")

        # LLM scores từ dataset gốc
        for col in config.SCORE_COLS:
            slug = _slugify(col)
            record[f"llm_{slug}"] = row.get(f"llm_{slug}")

        # Improved scores
        if final_scores:
            for col in config.SCORE_COLS:
                slug = _slugify(col)
                record[f"improved_{slug}"] = final_scores.get(col)
            log.info("  → Scores: %s", final_scores)
        else:
            for col in config.SCORE_COLS:
                slug = _slugify(col)
                record[f"improved_{slug}"] = None
            log.warning("  → Failed to get scores")

        results.append(record)

        # Rate limit sleep (không sleep sau sample cuối cùng)
        if idx < len(sample) - 1:
            log.info("  Sleep %ds (rate limit)...", config.RATE_LIMIT_SLEEP)
            time.sleep(config.RATE_LIMIT_SLEEP)

    log.info("Tổng số API calls: %d", total_calls)

    result_df = pd.DataFrame(results)

    # Tính metrics improved
    human_cols   = [f"human_avg_{_slugify(c)}" for c in config.SCORE_COLS]
    improved_cols = [f"improved_{_slugify(c)}" for c in config.SCORE_COLS]

    valid = [
        (h, im) for h, im in zip(human_cols, improved_cols)
        if h in result_df.columns and im in result_df.columns
    ]

    if valid:
        log.info("\n--- Metrics per criterion (Improved Judge) ---")
        per_criterion = compute_metrics_per_criterion(
            result_df,
            [v[0] for v in valid],
            [v[1] for v in valid],
        )
        log.info("\n%s", per_criterion.to_string())

        # Overall metrics
        all_true, all_pred = [], []
        for hc, ic in valid:
            sub = result_df[[hc, ic]].dropna()
            all_true.extend(sub[hc].values)
            all_pred.extend(sub[ic].values)

        overall = compute_all_metrics(
            y_true=np.array(all_true),
            y_pred=np.array(all_pred),
            prefix="overall",
        )
        log.info("\n--- Overall Metrics (Improved) ---")
        for k, v in overall.items():
            log.info("  %-30s : %.4f", k, v)

    # Lưu kết quả
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    log.info("Đã lưu improved results → %s", save_path)

    return result_df


if __name__ == "__main__":
    from data_merger import load_merged
    merged = load_merged()
    result = run_improved_evaluation(merged)
    print("\nImproved results (mẫu đầu):")
    print(result.head(3).to_string())
