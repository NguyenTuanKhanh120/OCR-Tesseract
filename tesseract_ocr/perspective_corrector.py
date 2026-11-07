"""Chỉnh phối cảnh từng panel đã cắt từ trang PDF.

Đây chỉ là bước 3 của pipeline OCR theo vùng. Module tìm bốn cạnh dài gần
mép panel, tính bốn góc, kiểm tra hình học rồi biến đổi phối cảnh về hình chữ
nhật. Kết quả luôn kèm overlay và manifest để kiểm tra.

Không loại logo/QR/barcode, không tiền xử lý nền và không chạy OCR.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


Point = Tuple[float, float]
Segment = Tuple[int, int, int, int]


class PerspectiveDetectionError(RuntimeError):
    """Không thể xác định tứ giác panel đủ an toàn để chỉnh phối cảnh."""


class DetectedQuad(NamedTuple):
    points: np.ndarray
    confidence: float
    area_ratio: float
    max_tilt_degrees: float
    segments: Dict[str, Segment]


def segment_length(segment: Segment) -> float:
    x1, y1, x2, y2 = segment
    return math.hypot(x2 - x1, y2 - y1)


def segment_angle_degrees(segment: Segment) -> float:
    x1, y1, x2, y2 = segment
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def line_from_segment(segment: Segment) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in segment]
    line = np.array([y1 - y2, x2 - x1, (x1 * y2) - (x2 * y1)], dtype=np.float64)
    normal = math.hypot(line[0], line[1])
    if normal <= 1e-9:
        raise PerspectiveDetectionError("Phát hiện đoạn thẳng có độ dài bằng 0")
    return line / normal


def line_intersection(first: np.ndarray, second: np.ndarray) -> Point:
    determinant = (first[0] * second[1]) - (second[0] * first[1])
    if abs(determinant) <= 1e-8:
        raise PerspectiveDetectionError("Hai cạnh panel gần song song, không có giao điểm ổn định")
    x = ((first[1] * second[2]) - (second[1] * first[2])) / determinant
    y = ((first[2] * second[0]) - (second[2] * first[0])) / determinant
    return float(x), float(y)


def coordinate_at_center(segment: Segment, orientation: str, width: int, height: int) -> float:
    x1, y1, x2, y2 = [float(value) for value in segment]
    if orientation == "horizontal":
        center_x = (width - 1) / 2.0
        if abs(x2 - x1) <= 1e-8:
            return (y1 + y2) / 2.0
        return y1 + ((center_x - x1) * (y2 - y1) / (x2 - x1))
    center_y = (height - 1) / 2.0
    if abs(y2 - y1) <= 1e-8:
        return (x1 + x2) / 2.0
    return x1 + ((center_y - y1) * (x2 - x1) / (y2 - y1))


def select_outer_segment(
    candidates: Sequence[Segment],
    side: str,
    width: int,
    height: int,
) -> Segment:
    if not candidates:
        raise PerspectiveDetectionError(f"Không tìm thấy cạnh {side} của panel")
    longest = max(segment_length(segment) for segment in candidates)
    strong = [
        segment for segment in candidates if segment_length(segment) >= longest * 0.68
    ]
    orientation = "horizontal" if side in ("top", "bottom") else "vertical"
    reverse = side in ("bottom", "right")
    return sorted(
        strong,
        key=lambda segment: coordinate_at_center(
            segment, orientation, width, height
        ),
        reverse=reverse,
    )[0]


def collect_border_segments(image: np.ndarray) -> Dict[str, List[Segment]]:
    if image is None or image.size == 0:
        raise ValueError("Ảnh panel đầu vào rỗng")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    if height < 120 or width < 120:
        raise ValueError(f"Ảnh panel quá nhỏ: {width}x{height}")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 150)
    _, dark = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY_INV)
    horizontal_mask = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, width // 4), 1)),
    )
    vertical_mask = cv2.morphologyEx(
        dark,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, height // 4))),
    )
    line_source = cv2.bitwise_or(edges, horizontal_mask)
    line_source = cv2.bitwise_or(line_source, vertical_mask)

    lines = cv2.HoughLinesP(
        line_source,
        1,
        np.pi / 720.0,
        threshold=max(60, min(width, height) // 10),
        minLineLength=max(80, min(width, height) * 2 // 5),
        maxLineGap=max(30, min(width, height) // 18),
    )
    if lines is None:
        raise PerspectiveDetectionError("Không phát hiện được đường viền panel")

    groups: Dict[str, List[Segment]] = {
        "top": [],
        "bottom": [],
        "left": [],
        "right": [],
    }
    for raw_segment in np.asarray(lines).reshape(-1, 4):
        segment = tuple(int(value) for value in raw_segment)
        x1, y1, x2, y2 = segment
        length = segment_length(segment)
        angle = abs(segment_angle_degrees(segment)) % 180.0
        horizontal_deviation = min(angle, abs(180.0 - angle))
        vertical_deviation = abs(90.0 - angle)
        midpoint_x = (x1 + x2) / 2.0
        midpoint_y = (y1 + y2) / 2.0

        if horizontal_deviation <= 12.0 and length >= width * 0.45:
            if midpoint_y <= height * 0.18:
                groups["top"].append(segment)
            if midpoint_y >= height * 0.82:
                groups["bottom"].append(segment)
        if vertical_deviation <= 12.0 and length >= height * 0.45:
            if midpoint_x <= width * 0.18:
                groups["left"].append(segment)
            if midpoint_x >= width * 0.82:
                groups["right"].append(segment)
    return groups


def detect_panel_quad(image: np.ndarray) -> DetectedQuad:
    """Phát hiện tứ giác đường viền ngoài của một panel."""
    height, width = image.shape[:2]
    groups = collect_border_segments(image)
    selected = {
        side: select_outer_segment(candidates, side, width, height)
        for side, candidates in groups.items()
    }
    lines = {side: line_from_segment(segment) for side, segment in selected.items()}
    points = np.array(
        [
            line_intersection(lines["top"], lines["left"]),
            line_intersection(lines["top"], lines["right"]),
            line_intersection(lines["bottom"], lines["right"]),
            line_intersection(lines["bottom"], lines["left"]),
        ],
        dtype=np.float32,
    )

    if not np.isfinite(points).all():
        raise PerspectiveDetectionError("Bốn góc panel chứa giá trị không hợp lệ")
    margin_x = width * 0.10
    margin_y = height * 0.10
    if (
        np.any(points[:, 0] < -margin_x)
        or np.any(points[:, 0] > (width - 1) + margin_x)
        or np.any(points[:, 1] < -margin_y)
        or np.any(points[:, 1] > (height - 1) + margin_y)
    ):
        raise PerspectiveDetectionError("Giao điểm đường viền nằm quá xa ảnh panel")

    contour = points.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise PerspectiveDetectionError("Bốn góc panel không tạo thành tứ giác lồi")
    area_ratio = abs(cv2.contourArea(contour)) / float(width * height)
    if not 0.55 <= area_ratio <= 1.10:
        raise PerspectiveDetectionError(
            f"Diện tích tứ giác panel không hợp lệ: {area_ratio:.3f}"
        )

    top_width = float(np.linalg.norm(points[1] - points[0]))
    bottom_width = float(np.linalg.norm(points[2] - points[3]))
    left_height = float(np.linalg.norm(points[3] - points[0]))
    right_height = float(np.linalg.norm(points[2] - points[1]))
    if min(top_width, bottom_width) < width * 0.60:
        raise PerspectiveDetectionError("Cạnh ngang panel quá ngắn")
    if min(left_height, right_height) < height * 0.60:
        raise PerspectiveDetectionError("Cạnh dọc panel quá ngắn")

    deviations = []
    for side, segment in selected.items():
        angle = abs(segment_angle_degrees(segment)) % 180.0
        if side in ("top", "bottom"):
            deviations.append(min(angle, abs(180.0 - angle)))
        else:
            deviations.append(abs(90.0 - angle))
    max_tilt = max(deviations)
    support_ratios = [
        segment_length(selected["top"]) / width,
        segment_length(selected["bottom"]) / width,
        segment_length(selected["left"]) / height,
        segment_length(selected["right"]) / height,
    ]
    support_score = min(1.0, min(support_ratios) / 0.75)
    area_score = min(1.0, area_ratio / 0.82)
    tilt_score = max(0.0, 1.0 - (max_tilt / 12.0))
    confidence = (0.50 * support_score) + (0.35 * area_score) + (0.15 * tilt_score)

    return DetectedQuad(
        points=points,
        confidence=round(float(confidence), 4),
        area_ratio=round(float(area_ratio), 4),
        max_tilt_degrees=round(float(max_tilt), 3),
        segments=selected,
    )


def rectify_from_quad(
    image: np.ndarray,
    points: np.ndarray,
    output_margin: int = 8,
) -> np.ndarray:
    if output_margin < 0:
        raise ValueError("output_margin không được âm")
    top_left, top_right, bottom_right, bottom_left = points.astype(np.float32)
    panel_width = int(
        round(
            max(
                np.linalg.norm(top_right - top_left),
                np.linalg.norm(bottom_right - bottom_left),
            )
        )
    )
    panel_height = int(
        round(
            max(
                np.linalg.norm(bottom_left - top_left),
                np.linalg.norm(bottom_right - top_right),
            )
        )
    )
    if panel_width < 50 or panel_height < 50:
        raise PerspectiveDetectionError(
            f"Kích thước sau chỉnh quá nhỏ: {panel_width}x{panel_height}"
        )

    output_width = panel_width + (2 * output_margin)
    output_height = panel_height + (2 * output_margin)
    destination = np.array(
        [
            [output_margin, output_margin],
            [output_margin + panel_width - 1, output_margin],
            [output_margin + panel_width - 1, output_margin + panel_height - 1],
            [output_margin, output_margin + panel_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points.astype(np.float32), destination)
    return cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def make_quad_overlay(image: np.ndarray, detected: DetectedQuad) -> np.ndarray:
    overlay = image.copy()
    polygon = np.round(detected.points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [polygon], True, (0, 180, 0), 6, cv2.LINE_AA)
    labels = ("TL", "TR", "BR", "BL")
    for label, point in zip(labels, detected.points):
        x, y = [int(round(value)) for value in point]
        cv2.circle(overlay, (x, y), 13, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            label,
            (x + 14, max(30, y - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"confidence={detected.confidence:.3f}  max_tilt={detected.max_tilt_degrees:.2f} deg",
        (25, max(45, image.shape[0] - 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 120, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def points_as_json(points: np.ndarray) -> Dict[str, Dict[str, float]]:
    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    return {
        name: {"x": round(float(point[0]), 3), "y": round(float(point[1]), 3)}
        for name, point in zip(names, points)
    }


def rectify_panels(
    panels_dir: Path,
    output_dir: Path,
    min_confidence: float = 0.65,
    output_margin: int = 8,
) -> Dict[str, Any]:
    """Chỉnh phối cảnh mọi panel được mô tả trong panels_manifest.json."""
    panels_dir = Path(panels_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = panels_dir / "panels_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy panels manifest: {manifest_path}")
    if output_dir == panels_dir or output_dir in panels_dir.parents:
        raise ValueError("Thư mục ảnh chỉnh phối cảnh phải tách khỏi thư mục panel")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence phải nằm trong khoảng 0..1")
    if output_margin < 0:
        raise ValueError("output_margin không được âm")

    panels_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_specs = panels_manifest.get("pages", [])
    if not page_specs:
        raise RuntimeError("Panels manifest không có trang")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    page_results: List[Dict[str, Any]] = []
    successful_panels = 0
    needs_review_panels = 0

    for page_spec in page_specs:
        page_number = int(page_spec["page_number"])
        page_dir = output_dir / f"page_{page_number:04d}"
        panel_results: List[Dict[str, Any]] = []
        for panel_spec in page_spec.get("panels", []):
            panel_index = int(panel_spec["panel_index"])
            source_path = panels_dir / str(panel_spec["file"])
            image = cv2.imdecode(
                np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"Không thể đọc ảnh panel: {source_path}")
            source_height, source_width = image.shape[:2]
            stem = f"panel_{panel_index:02d}"
            rectified_name = f"{stem}_rectified.png"
            overlay_name = f"{stem}_perspective_overlay.png"
            rectified_path = page_dir / rectified_name
            overlay_path = page_dir / overlay_name

            try:
                detected = detect_panel_quad(image)
                rectified = rectify_from_quad(
                    image, detected.points, output_margin=output_margin
                )
                overlay = make_quad_overlay(image, detected)
                status = (
                    "success"
                    if detected.confidence >= min_confidence
                    else "needs_review"
                )
                detection_error: Optional[str] = None
                corners: Optional[Dict[str, Dict[str, float]]] = points_as_json(
                    detected.points
                )
                confidence = detected.confidence
                area_ratio = detected.area_ratio
                max_tilt_degrees = detected.max_tilt_degrees
            except PerspectiveDetectionError as error:
                rectified = image.copy()
                overlay = image.copy()
                cv2.putText(
                    overlay,
                    f"NEEDS REVIEW: {error}",
                    (25, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                status = "needs_review"
                detection_error = str(error)
                corners = None
                confidence = 0.0
                area_ratio = 0.0
                max_tilt_degrees = 0.0

            save_png(rectified_path, rectified)
            save_png(overlay_path, overlay)
            output_height, output_width = rectified.shape[:2]
            if status == "success":
                successful_panels += 1
            else:
                needs_review_panels += 1
            panel_results.append(
                {
                    "panel_index": panel_index,
                    "source_file": str(panel_spec["file"]),
                    "source_width": source_width,
                    "source_height": source_height,
                    "rectified_file": str(
                        Path(f"page_{page_number:04d}") / rectified_name
                    ).replace("\\", "/"),
                    "overlay_file": str(
                        Path(f"page_{page_number:04d}") / overlay_name
                    ).replace("\\", "/"),
                    "output_width": output_width,
                    "output_height": output_height,
                    "status": status,
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                    "max_tilt_degrees": max_tilt_degrees,
                    "corners": corners,
                    "detection_error": detection_error,
                    "png_bytes": rectified_path.stat().st_size,
                    "png_sha256": sha256_file(rectified_path),
                }
            )

        page_status = (
            "success"
            if panel_results and all(item["status"] == "success" for item in panel_results)
            else "needs_review"
        )
        page_results.append(
            {
                "page_number": page_number,
                "status": page_status,
                "panel_count": len(panel_results),
                "panels": panel_results,
            }
        )

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "rectify_panel_perspective",
        "algorithm": "edge_hough_quad_v1",
        "source_panels_manifest": str(manifest_path),
        "source_panels_manifest_sha256": sha256_file(manifest_path),
        "min_confidence": min_confidence,
        "output_margin_pixels": output_margin,
        "page_count": len(page_results),
        "panel_count": successful_panels + needs_review_panels,
        "successful_panels": successful_panels,
        "needs_review_panels": needs_review_panels,
        "processing_seconds": round(time.perf_counter() - started, 3),
        "pages": page_results,
    }
    atomic_write_json(output_dir / "perspective_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chỉnh phối cảnh các panel đã cắt; chưa tiền xử lý và chưa OCR."
    )
    parser.add_argument("--panels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--output-margin", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = rectify_panels(
        args.panels_dir,
        args.output_dir,
        min_confidence=args.min_confidence,
        output_margin=args.output_margin,
    )
    print(
        f"Đã chỉnh {result['panel_count']} panel: "
        f"{result['successful_panels']} thành công, "
        f"{result['needs_review_panels']} cần kiểm tra"
    )
    for page in result["pages"]:
        details = ", ".join(
            f"panel {panel['panel_index']}={panel['status']} "
            f"(confidence {panel['confidence']:.3f}, tilt {panel['max_tilt_degrees']:.2f}°)"
            for panel in page["panels"]
        )
        print(f"Trang {page['page_number']}: {details}")
    print(str(Path(args.output_dir).resolve() / "perspective_manifest.json"))
    return 0 if result["needs_review_panels"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
