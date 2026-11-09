"""Chấm điểm và chọn giữa OCR hiện tại với biến thể đã tiền xử lý."""

from __future__ import annotations

import math
from typing import Any, Dict


def ocr_quality_score(
    confidence: float,
    text_metrics: Dict[str, Any],
    word_count: int,
    reference_word_count: int | None = None,
) -> float:
    """Điểm tương đối dùng trên cùng một trang, không phải độ chính xác."""
    suspicious = float(text_metrics.get("suspicious_token_ratio", 0.0))
    symbol_ratio = float(text_metrics.get("symbol_character_ratio", 0.0))
    non_latin = float(text_metrics.get("non_latin_letter_ratio", 0.0))
    score = (
        float(confidence)
        - (0.55 * suspicious)
        - (0.15 * symbol_ratio)
        - (0.10 * non_latin)
    )
    if reference_word_count and reference_word_count > 0 and word_count > 0:
        ratio = float(word_count) / float(reference_word_count)
        score -= 0.04 * abs(math.log(max(ratio, 1e-6)))
    return float(score)


def select_ocr_variant(
    current: Dict[str, Any],
    candidate: Dict[str, Any],
    pixel_difference_ratio: float,
    minimum_score_gain: float = 0.015,
) -> Dict[str, Any]:
    """Chọn ứng viên khi điểm tăng đủ và không vi phạm guardrail."""
    if not 0.0 <= pixel_difference_ratio <= 1.0:
        raise ValueError("pixel_difference_ratio phải nằm trong khoảng 0..1")
    if minimum_score_gain < 0:
        raise ValueError("minimum_score_gain không được âm")

    current_confidence = float(current["confidence"])
    candidate_confidence = float(candidate["confidence"])
    current_words = int(current["word_count"])
    candidate_words = int(candidate["word_count"])
    current_metrics = dict(current["text_metrics"])
    candidate_metrics = dict(candidate["text_metrics"])
    current_suspicious = float(current_metrics.get("suspicious_token_ratio", 0.0))
    candidate_suspicious = float(
        candidate_metrics.get("suspicious_token_ratio", 0.0)
    )
    confidence_delta = candidate_confidence - current_confidence
    suspicious_delta = candidate_suspicious - current_suspicious

    current_score = ocr_quality_score(
        current_confidence,
        current_metrics,
        current_words,
    )
    candidate_score = ocr_quality_score(
        candidate_confidence,
        candidate_metrics,
        candidate_words,
        reference_word_count=current_words,
    )
    score_gain = candidate_score - current_score

    common = {
        "current_score": round(current_score, 6),
        "candidate_score": round(candidate_score, 6),
        "score_gain": round(score_gain, 6),
        "minimum_score_gain": minimum_score_gain,
        "confidence_delta": round(confidence_delta, 6),
        "suspicious_token_delta": round(suspicious_delta, 6),
        "pixel_difference_ratio": round(pixel_difference_ratio, 6),
    }
    if pixel_difference_ratio <= 1e-6:
        return {
            **common,
            "selected": "current",
            "reason": "identical_pixels_keep_current",
            "guardrail_passed": True,
        }

    minimum_words = max(5, int(round(current_words * 0.40)))
    if candidate_words < minimum_words:
        return {
            **common,
            "selected": "current",
            "reason": "candidate_text_too_short",
            "guardrail_passed": False,
        }
    if confidence_delta < -0.03 and suspicious_delta > -0.05:
        return {
            **common,
            "selected": "current",
            "reason": "candidate_confidence_drop",
            "guardrail_passed": False,
        }
    if suspicious_delta > 0.04 and confidence_delta < 0.06:
        return {
            **common,
            "selected": "current",
            "reason": "candidate_suspicious_tokens_increased",
            "guardrail_passed": False,
        }
    if score_gain >= minimum_score_gain:
        return {
            **common,
            "selected": "candidate",
            "reason": "candidate_quality_score_improved",
            "guardrail_passed": True,
        }
    return {
        **common,
        "selected": "current",
        "reason": "candidate_gain_below_threshold",
        "guardrail_passed": True,
    }
