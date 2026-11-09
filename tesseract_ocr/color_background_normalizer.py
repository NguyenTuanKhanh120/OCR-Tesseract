"""Trung hòa nền màu tài liệu bằng độ lệch màu cục bộ trong không gian LAB.

Thuật toán ước lượng màu nền chậm quanh từng pixel, đo khoảng cách LAB giữa
pixel và nền, rồi ánh xạ mọi nét khác nền thành nét tối trên nền trắng. Nhờ
đó giữ được cả chữ đen trên nền xanh/vàng và chữ vàng trên nền đỏ.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np


def _validate_bgr(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Trung hòa nền màu yêu cầu ảnh BGR ba kênh")
    return image[:, :, :3]


def detect_colored_background(
    image: np.ndarray,
    chroma_p99_threshold: float = 20.0,
    colored_area_threshold: float = 0.04,
) -> Dict[str, Any]:
    """Phát hiện nền màu diện rộng, tránh nhầm con dấu/chữ màu cục bộ."""
    image = _validate_bgr(image)
    if chroma_p99_threshold < 0 or not 0.0 <= colored_area_threshold <= 1.0:
        raise ValueError("Ngưỡng phát hiện màu không hợp lệ")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = hsv[:, :, 2] >= 160
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt((a * a) + (b * b))
    bright_chroma = chroma[bright]
    chroma_p99 = (
        float(np.percentile(bright_chroma, 99)) if bright_chroma.size else 0.0
    )
    colored_area_ratio = float(np.mean(bright & (chroma >= 10.0)))
    detected = (
        chroma_p99 >= chroma_p99_threshold
        and colored_area_ratio >= colored_area_threshold
    )
    return {
        "detected": bool(detected),
        "bright_chroma_p99": round(chroma_p99, 4),
        "colored_bright_area_ratio": round(colored_area_ratio, 6),
        "chroma_p99_threshold": chroma_p99_threshold,
        "colored_area_threshold": colored_area_threshold,
    }


def prepare_color_document_geometry(
    image: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Áp dụng cùng khử đường màu/deskew của profile grayscale nhưng giữ BGR."""
    from tesseract_ocr.preprocessing import (
        detect_deskew_angle,
        remove_long_colored_lines,
        rotate_image_expand,
    )

    image = _validate_bgr(image)
    cleaned, colored_lines_removed = remove_long_colored_lines(image)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    angle = detect_deskew_angle(gray)
    if abs(angle) > 0.25:
        cleaned = rotate_image_expand(cleaned, angle)
    return cleaned, {
        "colored_lines_removed": colored_lines_removed,
        "deskew_angle": round(float(angle), 4) if abs(angle) > 0.25 else 0.0,
    }


def estimate_local_lab_background(
    image: np.ndarray,
    working_max_dimension: int = 1400,
    median_ratio: float = 0.015,
    blur_ratio: float = 0.055,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Ước lượng nền LAB chậm, bỏ qua nét chữ nhỏ ở cả hai cực sáng/tối."""
    image = _validate_bgr(image)
    if working_max_dimension < 128:
        raise ValueError("working_max_dimension phải từ 128 trở lên")
    if median_ratio <= 0 or blur_ratio <= 0:
        raise ValueError("median_ratio và blur_ratio phải lớn hơn 0")

    height, width = image.shape[:2]
    scale = min(1.0, float(working_max_dimension) / max(height, width))
    working = (
        cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else image
    )
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    short_side = min(lab.shape[:2])
    median_size = max(5, int(round(short_side * median_ratio)))
    median_size += 1 - (median_size % 2)
    median_size = min(median_size, 31)
    blur_size = max(7, int(round(short_side * blur_ratio)))
    blur_size += 1 - (blur_size % 2)

    background_channels = []
    for channel in cv2.split(lab):
        smoothed = cv2.medianBlur(channel, median_size)
        background_channels.append(
            cv2.GaussianBlur(
                smoothed,
                (blur_size, blur_size),
                sigmaX=0,
                borderType=cv2.BORDER_REPLICATE,
            )
        )
    background_small = cv2.merge(background_channels)
    background = (
        cv2.resize(
            background_small,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        if background_small.shape[:2] != (height, width)
        else background_small
    )
    return background, {
        "working_scale": round(scale, 6),
        "working_width": int(working.shape[1]),
        "working_height": int(working.shape[0]),
        "median_kernel": median_size,
        "blur_kernel": blur_size,
    }


def color_difference_map(
    image: np.ndarray,
    background_lab: np.ndarray,
    chroma_weight: float = 1.15,
) -> np.ndarray:
    """Tính độ khác nền, tăng nhẹ trọng số khác biệt màu a/b."""
    image = _validate_bgr(image)
    if background_lab.shape != image.shape:
        raise ValueError("Ảnh và trường nền LAB phải cùng kích thước")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = background_lab.astype(np.float32)
    delta_l = lab[:, :, 0] - background[:, :, 0]
    delta_a = lab[:, :, 1] - background[:, :, 1]
    delta_b = lab[:, :, 2] - background[:, :, 2]
    return np.sqrt(
        (delta_l * delta_l)
        + float(chroma_weight) * ((delta_a * delta_a) + (delta_b * delta_b))
    )


def neutralize_color_background(
    image: np.ndarray,
    noise_floor: float = 15.0,
    full_contrast_delta: float = 60.0,
    chroma_weight: float = 1.15,
    working_max_dimension: int = 1400,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Đưa nền màu về trắng và mọi nét khác nền về thang xám tối."""
    image = _validate_bgr(image)
    if noise_floor < 0:
        raise ValueError("noise_floor không được âm")
    if full_contrast_delta <= noise_floor:
        raise ValueError("full_contrast_delta phải lớn hơn noise_floor")
    if chroma_weight <= 0:
        raise ValueError("chroma_weight phải lớn hơn 0")

    background, field_meta = estimate_local_lab_background(
        image, working_max_dimension=working_max_dimension
    )
    difference = color_difference_map(
        image, background, chroma_weight=chroma_weight
    )
    ink_strength = np.clip(
        (difference - float(noise_floor))
        / (float(full_contrast_delta) - float(noise_floor)),
        0.0,
        1.0,
    )
    neutral = np.clip(np.rint(255.0 * (1.0 - ink_strength)), 0, 255).astype(
        np.uint8
    )

    metadata: Dict[str, Any] = {
        "applied": True,
        "algorithm": "local_lab_color_difference_v1",
        "noise_floor": noise_floor,
        "full_contrast_delta": full_contrast_delta,
        "chroma_weight": chroma_weight,
        "difference_median": round(float(np.median(difference)), 4),
        "difference_p95": round(float(np.percentile(difference, 95)), 4),
        "difference_p99": round(float(np.percentile(difference, 99)), 4),
        "dark_output_ratio": round(float(np.mean(neutral < 128)), 6),
        "near_white_ratio": round(float(np.mean(neutral >= 245)), 6),
        **field_meta,
    }
    return neutral, metadata
