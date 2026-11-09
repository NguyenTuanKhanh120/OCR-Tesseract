"""Tích hợp cân bằng sáng, trung hòa màu và chọn biến thể cho một trang OCR."""

from __future__ import annotations

import time
from typing import Any, Dict

import cv2
import numpy as np

from tesseract_ocr.background_baseline import text_quality_metrics
from tesseract_ocr.color_background_normalizer import (
    detect_colored_background,
    neutralize_color_background,
    prepare_color_document_geometry,
)
from tesseract_ocr.config import TesseractConfig
from tesseract_ocr.illumination_normalizer import normalize_illumination
from tesseract_ocr.ocr_variant_selector import select_ocr_variant
from tesseract_ocr.preprocessing import preprocess_image
from tesseract_ocr.tesseract_engine import run_tesseract_on_image


def _to_gray(image: np.ndarray) -> np.ndarray:
    return (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )


def _variant(result: Any) -> Dict[str, Any]:
    return {
        "confidence": float(result.confidence),
        "word_count": len(result.boxes),
        "text_metrics": text_quality_metrics(result.text),
    }


def _disabled_decision() -> Dict[str, Any]:
    return {
        "selected": "current",
        "reason": "adaptive_background_disabled",
        "guardrail_passed": True,
        "current_score": 0.0,
        "candidate_score": 0.0,
        "score_gain": 0.0,
        "minimum_score_gain": 0.015,
        "confidence_delta": 0.0,
        "suspicious_token_delta": 0.0,
        "pixel_difference_ratio": 0.0,
    }


def run_adaptive_background_ocr(
    raw_image: np.ndarray,
    config: TesseractConfig,
    page_number: int,
    preprocessing_mode: str,
    enabled: bool,
    minimum_score_gain: float = 0.015,
) -> Dict[str, Any]:
    """OCR một trang và chỉ chọn biến thể xử lý khi guardrail cho phép."""
    started = time.perf_counter()
    current_image, current_meta = preprocess_image(
        raw_image, mode=preprocessing_mode
    )
    current_result = run_tesseract_on_image(
        current_image,
        config,
        page_number=page_number,
        prep_meta=current_meta.to_dict(),
    )
    if not enabled or current_result.status == "error":
        decision = _disabled_decision()
        if current_result.status == "error":
            decision["reason"] = "current_ocr_error_no_candidate"
        return {
            "selected_image": current_image,
            "selected_result": current_result,
            "selected_variant": "current",
            "decision": decision,
            "candidate_kind": "none",
            "color_detection": None,
            "candidate_result": None,
            "processing_seconds": round(time.perf_counter() - started, 3),
        }

    color_detection = detect_colored_background(raw_image)
    candidate_kind = "color_normalized" if color_detection["detected"] else "illumination_balanced"
    candidate_metadata: Dict[str, Any]
    if color_detection["detected"]:
        prepared_color, geometry_meta = prepare_color_document_geometry(raw_image)
        candidate_image, normalization_meta = neutralize_color_background(
            prepared_color
        )
        candidate_metadata = {
            "geometry": geometry_meta,
            "color_normalization": normalization_meta,
        }
    else:
        candidate_image, illumination_meta = normalize_illumination(
            current_image
        )
        candidate_metadata = {"illumination": illumination_meta}

    current_gray = _to_gray(current_image)
    candidate_gray = _to_gray(candidate_image)
    if candidate_gray.shape != current_gray.shape:
        return {
            "selected_image": current_image,
            "selected_result": current_result,
            "selected_variant": "current",
            "decision": {
                **_disabled_decision(),
                "reason": "candidate_geometry_mismatch_keep_current",
            },
            "candidate_kind": candidate_kind,
            "color_detection": color_detection,
            "candidate_result": None,
            "candidate_metadata": candidate_metadata,
            "processing_seconds": round(time.perf_counter() - started, 3),
        }

    pixel_difference_ratio = float(np.mean(current_gray != candidate_gray))
    if pixel_difference_ratio <= 1e-6:
        decision = select_ocr_variant(
            _variant(current_result),
            _variant(current_result),
            pixel_difference_ratio=0.0,
            minimum_score_gain=minimum_score_gain,
        )
        return {
            "selected_image": current_image,
            "selected_result": current_result,
            "selected_variant": "current",
            "decision": decision,
            "candidate_kind": candidate_kind,
            "color_detection": color_detection,
            "candidate_result": None,
            "candidate_metadata": candidate_metadata,
            "processing_seconds": round(time.perf_counter() - started, 3),
        }

    candidate_prep_meta = {
        **current_meta.to_dict(),
        "adaptive_background": {
            "candidate_kind": candidate_kind,
            **candidate_metadata,
        },
    }
    candidate_result = run_tesseract_on_image(
        candidate_image,
        config,
        page_number=page_number,
        prep_meta=candidate_prep_meta,
    )
    decision = select_ocr_variant(
        _variant(current_result),
        _variant(candidate_result),
        pixel_difference_ratio=pixel_difference_ratio,
        minimum_score_gain=minimum_score_gain,
    )
    choose_candidate = decision["selected"] == "candidate"
    selected_result = candidate_result if choose_candidate else current_result
    selected_image = candidate_image if choose_candidate else current_image
    selected_result.preprocessing = {
        **dict(selected_result.preprocessing),
        "adaptive_background": {
            "enabled": True,
            "selected_variant": "processed" if choose_candidate else "current",
            "candidate_kind": candidate_kind,
            "color_detection": color_detection,
            "decision": decision,
        },
    }
    return {
        "selected_image": selected_image,
        "selected_result": selected_result,
        "selected_variant": "processed" if choose_candidate else "current",
        "decision": decision,
        "candidate_kind": candidate_kind,
        "color_detection": color_detection,
        "candidate_result": candidate_result,
        "candidate_metadata": candidate_metadata,
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
