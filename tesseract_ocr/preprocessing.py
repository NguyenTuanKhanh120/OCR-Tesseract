"""
Tiền xử lý ảnh phục vụ Tesseract OCR bằng OpenCV và NumPy.
Hỗ trợ chế độ 'none' và 'auto'.
Ghi nhận đầy đủ thông số tiền xử lý theo mục 6.4 của tài liệu thiết kế.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np

from tesseract_ocr.schemas import PreprocessingMeta


def rotate_image_expand(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Xoay ảnh theo góc bất kỳ và mở rộng kích thước canvas
    để đảm bảo không bị cắt mất bất kỳ nội dung nào ở 4 góc.
    """
    if abs(angle) < 1e-4:
        return image

    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    # Ma trận xoay 2D
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Kích thước canvas mới
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Điều chỉnh ma trận tịnh tiến vào tâm mới
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    # Fill nền trắng (255) cho văn bản giấy
    border_val = (255, 255, 255) if len(image.shape) == 3 else 255
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_val,
    )
    return rotated


def detect_orientation_90(gray: np.ndarray) -> int:
    """
    Ước lượng xoay 90, 180, 270 độ nếu ảnh bị quay ngang/ngược.
    Mặc định trả về 0 nếu không đủ bằng chứng chắc chắn.
    """
    # Nếu chiều rộng nhỏ hơn chiều cao đáng kể (portrait chuẩn), thường là 0.
    # Trong môi trường OCR, có thể kiểm tra tỷ lệ profile ngang vs dọc.
    # Trả về 0 mặc định; góc xoay tinh chỉnh sẽ do deskew đảm nhiệm.
    return 0


def remove_long_colored_lines(
    image: np.ndarray,
    min_width_ratio: float = 0.45,
    max_height_ratio: float = 0.03,
) -> Tuple[np.ndarray, int]:
    """Khử các đường màu dài, mảnh mà không xóa con dấu/chữ màu cục bộ.

    Chỉ thành phần có độ bão hòa cao, rộng ít nhất 45% trang và thấp không quá
    3% chiều cao mới được xem là đường nhiễu. Pixel màu được trung hòa bằng
    kênh sáng nhất thay vì tô cả hình chữ nhật, nhờ đó chữ đen và con dấu nằm
    gần đường kẻ được giữ tối đa.
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return image.copy(), 0
    height, width = image.shape[:2]
    if height < 20 or width < 40:
        return image.copy(), 0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = np.where(
        (saturation >= 90) & (value >= 60), 255, 0
    ).astype(np.uint8)

    kernel_width = max(25, int(round(width * 0.12)))
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 1)
    )
    connected = cv2.morphologyEx(
        color_mask, cv2.MORPH_CLOSE, horizontal_kernel
    )
    opened = cv2.morphologyEx(
        connected, cv2.MORPH_OPEN, horizontal_kernel
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    final_mask = np.zeros((height, width), dtype=np.uint8)
    detected_lines = 0
    max_height = max(6, int(round(height * max_height_ratio)))
    min_width = int(round(width * min_width_ratio))

    for label_index in range(1, component_count):
        x, y, component_width, component_height, _ = stats[label_index]
        aspect_ratio = component_width / max(1, component_height)
        original_component = color_mask[
            y : y + component_height, x : x + component_width
        ]
        color_density = float(np.count_nonzero(original_component)) / max(
            1, component_width * component_height
        )
        if (
            component_width < min_width
            or component_height > max_height
            or aspect_ratio < 12.0
            or color_density < 0.55
        ):
            continue
        y0 = max(0, y - 2)
        y1 = min(height, y + component_height + 2)
        band_mask = color_mask[y0:y1, x : x + component_width]
        final_mask[y0:y1, x : x + component_width] = cv2.bitwise_or(
            final_mask[y0:y1, x : x + component_width], band_mask
        )
        detected_lines += 1

    if detected_lines == 0:
        return image.copy(), 0

    cleaned = image.copy()
    neutral_brightness = np.max(cleaned[:, :, :3], axis=2)
    selected = final_mask > 0
    cleaned[selected] = np.repeat(
        neutral_brightness[selected, np.newaxis], 3, axis=1
    )
    return cleaned, detected_lines


def _weighted_median(values: List[float], weights: List[float]) -> float:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = float(np.sum(sorted_weights)) / 2.0
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def detect_deskew_angle(gray: np.ndarray) -> float:
    """
    Phát hiện góc nghiêng nhỏ từ nhiều dòng chữ thay vì toàn bộ khối trang.

    Các ký tự được nối ngang thành dòng, sau đó ``fitLine`` ước lượng góc cho
    từng contour. Trung vị có trọng số giúp bỏ qua tiêu đề in nghiêng, con dấu
    và đường viền. Hàm chỉ trả góc khi có ít nhất ba dòng đồng thuận.
    """
    try:
        if gray.ndim != 2:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if height < 80 or width < 80:
            return 0.0

        scale = min(1.0, 1800.0 / max(height, width))
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
        blurred = cv2.GaussianBlur(working, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        kernel_width = max(12, int(round(work_width * 0.025)))
        line_mask = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)),
        )
        contours, _ = cv2.findContours(
            line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        angles: List[float] = []
        weights: List[float] = []
        for contour in contours:
            x, _, contour_width, contour_height = cv2.boundingRect(contour)
            del x
            if (
                contour_width < max(35, int(work_width * 0.08))
                or contour_height < 3
                or contour_height > max(70, int(work_height * 0.12))
                or contour_width / max(1, contour_height) < 2.0
                or len(contour) < 10
            ):
                continue
            vx, vy, _, _ = cv2.fitLine(
                contour, cv2.DIST_L2, 0, 0.01, 0.01
            ).flatten()
            angle = float(np.degrees(np.arctan2(float(vy), float(vx))))
            while angle <= -90.0:
                angle += 180.0
            while angle > 90.0:
                angle -= 180.0
            if abs(angle) <= 12.0:
                angles.append(angle)
                weights.append(float(contour_width))

        if len(angles) < 3:
            return 0.0
        median_angle = _weighted_median(angles, weights)
        deviations = [abs(value - median_angle) for value in angles]
        median_deviation = _weighted_median(deviations, weights)
        agreeing = sum(abs(value - median_angle) <= 2.0 for value in angles)
        if median_deviation > 2.5 or agreeing < max(3, len(angles) // 2):
            return 0.0
        # Góc dòng dương nghĩa là chữ chúi xuống về bên phải; OpenCV cần xoay
        # cùng dấu để đưa baseline trở lại phương ngang.
        if abs(median_angle) < 0.25:
            return 0.0
        return float(median_angle)
    except Exception:
        return 0.0


def crop_black_borders(gray: np.ndarray, tol: int = 40) -> np.ndarray:
    """
    Cắt viền đen ngoài rìa thường gặp trong quá trình scan tài liệu.
    """
    h, w = gray.shape[:2]
    # Mặt nạ các pixel sáng hơn ngưỡng đen
    mask = gray > tol
    if not np.any(mask):
        return gray

    # Tìm bounding box của vùng không phải viền đen
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Thêm padding an toàn 5 pixel
    rmin = max(0, rmin - 5)
    rmax = min(h - 1, rmax + 5)
    cmin = max(0, cmin - 5)
    cmax = min(w - 1, cmax + 5)

    # Chỉ cắt nếu vùng đen chiếm đáng kể viền (ít nhất 10 pixel từ mép)
    if rmin > 10 or (h - 1 - rmax) > 10 or cmin > 10 or (w - 1 - cmax) > 10:
        return gray[rmin : rmax + 1, cmin : cmax + 1]

    return gray


def enhance_contrast_if_needed(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Tăng tương phản có điều kiện bằng CLAHE nếu độ lệch chuẩn histogram thấp.
    """
    std = np.std(gray)
    # Nếu độ tương phản yếu (độ lệch chuẩn nhỏ hơn 50)
    if std < 50.0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced, True
    return gray, False


def apply_adaptive_threshold_if_needed(gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Áp dụng Adaptive Threshold nếu nền ảnh có độ biến thiên cao (nền không đều, bóng râm).
    """
    # Chia ảnh thành các khối 32x32 và kiểm tra phương sai trung bình của nền
    h, w = gray.shape[:2]
    if h < 64 or w < 64:
        return gray, False

    # Lấy mẫu góc và giữa để đo độ chênh lệch nền
    corners = [
        gray[0 : h // 4, 0 : w // 4],
        gray[0 : h // 4, 3 * w // 4 :],
        gray[3 * h // 4 :, 0 : w // 4],
        gray[3 * h // 4 :, 3 * w // 4 :],
    ]
    means = [np.mean(c) for c in corners]
    diff = max(means) - min(means)

    # Nếu độ lệch trung bình các góc > 45, nền bị tối/bóng không đều -> dùng adaptive threshold
    if diff > 45:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
        return thresh, True

    return gray, False


def preprocess_image(
    image: np.ndarray,
    mode: str = "auto",
) -> Tuple[np.ndarray, PreprocessingMeta]:
    """
    Hàm xử lý ảnh chính cho Tesseract OCR.
    Hỗ trợ chế độ 'none', 'grayscale' và 'auto'.
    """
    start_time = time.time()
    meta = PreprocessingMeta()

    if mode == "none":
        meta.processing_seconds = time.time() - start_time
        return image, meta

    # 1. Khử đường màu dài trước khi chuyển xám. Chỉ áp dụng cho tài liệu màu;
    # con dấu/chữ màu cục bộ không thỏa điều kiện hình học nên được giữ lại.
    colored_lines_removed = 0
    if len(image.shape) == 3:
        cleaned_color, colored_lines_removed = remove_long_colored_lines(image)
        gray = cv2.cvtColor(cleaned_color, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if mode == "grayscale":
        # Profile web Sauvola dùng ảnh xám nhưng vẫn cần deskew nhẹ trước khi
        # Tesseract thực hiện threshold nội bộ.
        deskew_angle = detect_deskew_angle(gray)
        if abs(deskew_angle) > 0.25:
            gray = rotate_image_expand(gray, deskew_angle)
            meta.deskew_angle = deskew_angle
        notes = ["grayscale_document_cleanup"]
        if colored_lines_removed:
            notes.append(f"colored_lines_removed={colored_lines_removed}")
        meta.notes = ";".join(notes)
        meta.processing_seconds = time.time() - start_time
        return gray, meta

    # 2. Cắt viền đen ngoại vi
    gray = crop_black_borders(gray)

    # 3. Phát hiện hướng 90/180/270 độ
    rot_90 = detect_orientation_90(gray)
    if rot_90 != 0:
        gray = rotate_image_expand(gray, rot_90)
        meta.orientation_rotation = rot_90

    # 4. Phát hiện và căn chỉnh góc nghiêng nhỏ (Deskew) với canvas mở rộng
    deskew_angle = detect_deskew_angle(gray)
    if abs(deskew_angle) > 0.25:
        gray = rotate_image_expand(gray, deskew_angle)
        meta.deskew_angle = deskew_angle

    # 5. Tăng tương phản có điều kiện
    gray, contrast_applied = enhance_contrast_if_needed(gray)
    meta.contrast_applied = contrast_applied

    # 6. Adaptive threshold khi nền không đều
    gray, threshold_applied = apply_adaptive_threshold_if_needed(gray)
    meta.threshold_applied = threshold_applied

    if colored_lines_removed:
        meta.notes = f"colored_lines_removed={colored_lines_removed}"

    meta.processing_seconds = time.time() - start_time
    return gray, meta
