"""Cân bằng sáng-tối cục bộ cho ảnh tài liệu.

Bước này chỉ hiệu chỉnh độ sáng. Nó không chọn kênh màu, không nhị phân hóa
và không tự chọn giữa ảnh gốc với ảnh đã xử lý. Thuật toán ước lượng trường
sáng chậm bằng median filter + Gaussian blur, sau đó giữ lại độ lệch
cục bộ của nét chữ quanh một mức nền thống nhất.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError("Ảnh đầu vào phải là ảnh xám hoặc BGR")


def _block_background_values(gray: np.ndarray, grid_size: int = 8) -> np.ndarray:
    height, width = gray.shape[:2]
    values = []
    for row in range(grid_size):
        y0 = row * height // grid_size
        y1 = (row + 1) * height // grid_size
        for column in range(grid_size):
            x0 = column * width // grid_size
            x1 = (column + 1) * width // grid_size
            block = gray[y0:y1, x0:x1]
            if block.size:
                values.append(float(np.percentile(block, 85)))
    return np.asarray(values, dtype=np.float32)


def background_range(gray: np.ndarray) -> float:
    """Độ chênh nền giữa các vùng trang, dùng để bật xử lý có điều kiện."""
    values = _block_background_values(_as_gray(gray))
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def estimate_illumination_field(
    gray: np.ndarray,
    working_max_dimension: int = 1400,
    median_ratio: float = 0.015,
    blur_ratio: float = 0.055,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ước lượng trường sáng chậm, bỏ qua cả chữ tối và chữ sáng nhỏ."""
    gray = _as_gray(gray)
    if working_max_dimension < 128:
        raise ValueError("working_max_dimension phải từ 128 trở lên")
    if median_ratio <= 0 or blur_ratio <= 0:
        raise ValueError("median_ratio và blur_ratio phải lớn hơn 0")

    height, width = gray.shape[:2]
    scale = min(1.0, float(working_max_dimension) / max(height, width))
    working = (
        cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else gray
    )
    work_height, work_width = working.shape[:2]
    short_side = min(work_height, work_width)
    median_size = max(5, int(round(short_side * median_ratio)))
    median_size += 1 - (median_size % 2)
    median_size = min(median_size, 31)
    blur_size = max(7, int(round(short_side * blur_ratio)))
    blur_size += 1 - (blur_size % 2)

    smoothed = cv2.medianBlur(working, median_size)
    field_small = cv2.GaussianBlur(
        smoothed,
        (blur_size, blur_size),
        sigmaX=0,
        borderType=cv2.BORDER_REPLICATE,
    )
    field = (
        cv2.resize(field_small, (width, height), interpolation=cv2.INTER_CUBIC)
        if field_small.shape != gray.shape
        else field_small
    )
    metadata = {
        "working_scale": round(scale, 6),
        "working_width": work_width,
        "working_height": work_height,
        "median_kernel": median_size,
        "blur_kernel": blur_size,
    }
    return field, metadata


def _recenter_local_contrast(
    gray: np.ndarray,
    field: np.ndarray,
    normal_background: float = 245.0,
    reverse_background: float = 215.0,
    reverse_detail_threshold: float = 18.0,
    reverse_pixel_ratio: float = 0.001,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    detail = gray.astype(np.float32) - field.astype(np.float32)
    positive_ratio = float(np.mean(detail >= reverse_detail_threshold))
    reverse_content = positive_ratio >= reverse_pixel_ratio
    target = reverse_background if reverse_content else normal_background

    low = float(np.percentile(detail, 0.5))
    high = float(np.percentile(detail, 99.5))
    negative_gain = min(1.0, target / max(abs(low), 1.0))
    positive_gain = min(1.0, (255.0 - target) / max(high, 1.0))
    balanced = np.where(
        detail < 0.0,
        target + (detail * negative_gain),
        target + (detail * positive_gain),
    )
    balanced = np.clip(np.rint(balanced), 0, 255).astype(np.uint8)
    return balanced, {
        "reverse_content_detected": reverse_content,
        "positive_detail_ratio": round(positive_ratio, 6),
        "detail_p005": round(low, 4),
        "detail_p995": round(high, 4),
        "negative_gain": round(negative_gain, 6),
        "positive_gain": round(positive_gain, 6),
        "target_background": target,
    }


def normalize_illumination(
    image: np.ndarray,
    activation_range: float = 18.0,
    working_max_dimension: int = 1400,
    color_chroma_p99: float | None = None,
    color_defer_threshold: float = 20.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Cân bằng sáng có điều kiện và trả về đầy đủ metadata so sánh."""
    if activation_range < 0:
        raise ValueError("activation_range không được âm")
    if color_defer_threshold < 0:
        raise ValueError("color_defer_threshold không được âm")
    gray = _as_gray(image)
    before = background_range(gray)
    if color_chroma_p99 is not None and color_chroma_p99 >= color_defer_threshold:
        return gray, {
            "applied": False,
            "reason": "deferred_colored_background_to_step2",
            "activation_range": activation_range,
            "color_chroma_p99": round(float(color_chroma_p99), 4),
            "color_defer_threshold": color_defer_threshold,
            "background_range_before": round(before, 4),
            "background_range_after": round(before, 4),
            "background_reduction_ratio": 0.0,
        }
    if before < activation_range:
        return gray, {
            "applied": False,
            "reason": "background_range_below_threshold",
            "activation_range": activation_range,
            "background_range_before": round(before, 4),
            "background_range_after": round(before, 4),
            "background_reduction_ratio": 0.0,
        }

    field, field_meta = estimate_illumination_field(
        gray, working_max_dimension=working_max_dimension
    )
    balanced, contrast_meta = _recenter_local_contrast(gray, field)
    after = background_range(balanced)
    reduction = (before - after) / max(before, 1e-6)
    metadata: Dict[str, Any] = {
        "applied": True,
        "reason": "uneven_illumination_detected",
        "activation_range": activation_range,
        "background_range_before": round(before, 4),
        "background_range_after": round(after, 4),
        "background_reduction_ratio": round(reduction, 6),
        **field_meta,
        **contrast_meta,
    }
    return balanced, metadata


def thin_line_retention(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Đo tỷ lệ biên gốc còn xuất hiện gần cùng vị trí sau xử lý."""
    reference_gray = _as_gray(reference)
    candidate_gray = _as_gray(candidate)
    if reference_gray.shape != candidate_gray.shape:
        raise ValueError("Hai ảnh phải cùng kích thước")
    reference_edges = cv2.Canny(reference_gray, 50, 150) > 0
    if not np.any(reference_edges):
        return 1.0
    candidate_edges = cv2.Canny(candidate_gray, 50, 150)
    nearby_candidate = cv2.dilate(
        candidate_edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    return float(np.mean(nearby_candidate[reference_edges]))
