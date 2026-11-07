"""Che logo, QR và barcode khỏi panel trước khi OCR.

Đây chỉ là bước 4 của pipeline OCR theo vùng. Module đọc ảnh đã chỉnh phối
cảnh ở bước 3, phát hiện các vùng phi văn bản và tạo bản sao có nền trắng.
Ảnh nguồn không bị sửa; mỗi panel có overlay và manifest để kiểm tra.

Không xử lý con dấu, không tăng tương phản/nền và không chạy OCR.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


Box = Tuple[int, int, int, int]


class ArtifactRegion(NamedTuple):
    kind: str
    box: Box
    confidence: float
    detector: str


def box_area(box: Box) -> int:
    return max(0, box[2]) * max(0, box[3])


def box_iou(first: Box, second: Box) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union else 0.0


def polygon_to_box(points: np.ndarray, width: int, height: int) -> Optional[Box]:
    polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 4 or not np.isfinite(polygon).all():
        return None
    left = max(0, int(math.floor(float(np.min(polygon[:, 0])))))
    top = max(0, int(math.floor(float(np.min(polygon[:, 1])))))
    right = min(width, int(math.ceil(float(np.max(polygon[:, 0])))) + 1)
    bottom = min(height, int(math.ceil(float(np.max(polygon[:, 1])))) + 1)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def expand_box(box: Box, width: int, height: int, padding: int) -> Box:
    x, y, box_width, box_height = box
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(width, x + box_width + padding)
    bottom = min(height, y + box_height + padding)
    return left, top, right - left, bottom - top


def deduplicate_regions(
    regions: Iterable[ArtifactRegion], overlap_threshold: float = 0.50
) -> List[ArtifactRegion]:
    selected: List[ArtifactRegion] = []
    for region in sorted(regions, key=lambda item: item.confidence, reverse=True):
        duplicate = next(
            (
                existing
                for existing in selected
                if existing.kind == region.kind
                and box_iou(existing.box, region.box) >= overlap_threshold
            ),
            None,
        )
        if duplicate is None:
            selected.append(region)
    return selected


def qr_points(detector: Any, image: np.ndarray) -> List[np.ndarray]:
    points_list: List[np.ndarray] = []
    try:
        if hasattr(detector, "detectMulti"):
            detected, points = detector.detectMulti(image)
            if detected and points is not None:
                points_list.extend(np.asarray(points).reshape(-1, 4, 2))
        if not points_list:
            detected, points = detector.detect(image)
            if detected and points is not None:
                points_list.extend(np.asarray(points).reshape(-1, 4, 2))
    except cv2.error:
        return []
    return points_list


def detect_qr_regions(image: np.ndarray) -> List[ArtifactRegion]:
    """Dò QR trên ảnh màu và các bản nhị phân để chịu được scan nhạt."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants: List[Tuple[str, np.ndarray, float]] = [
        ("color", image, 0.99),
        ("gray", gray, 0.97),
    ]
    for threshold in (120, 140, 170, 200):
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        variants.append((f"binary_{threshold}", binary, 0.95))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("binary_otsu", otsu, 0.95))

    found: List[ArtifactRegion] = []
    for variant_name, variant, confidence in variants:
        detector = cv2.QRCodeDetector()
        for points in qr_points(detector, variant):
            box = polygon_to_box(points, width, height)
            if box is None:
                continue
            _, _, box_width, box_height = box
            aspect = box_width / float(box_height)
            if min(box_width, box_height) < 30 or not 0.65 <= aspect <= 1.45:
                continue
            found.append(
                ArtifactRegion(
                    kind="qr",
                    box=box,
                    confidence=confidence,
                    detector=f"opencv_qr_{variant_name}",
                )
            )
    return deduplicate_regions(found, overlap_threshold=0.40)


def opencv_barcode_regions(image: np.ndarray) -> List[ArtifactRegion]:
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        return []
    height, width = image.shape[:2]
    try:
        detector = cv2.barcode_BarcodeDetector()
        detected, points = detector.detect(image)
    except (cv2.error, AttributeError):
        return []
    if not detected or points is None:
        return []

    regions: List[ArtifactRegion] = []
    for polygon in np.asarray(points).reshape(-1, 4, 2):
        box = polygon_to_box(polygon, width, height)
        if box is None:
            continue
        _, _, box_width, box_height = box
        aspect = max(box_width, box_height) / float(min(box_width, box_height))
        if aspect < 1.60:
            continue
        regions.append(
            ArtifactRegion(
                kind="barcode",
                box=box,
                confidence=0.98,
                detector="opencv_barcode",
            )
        )
    return regions


def barcode_line_score(binary_crop: np.ndarray) -> Tuple[float, int]:
    crop_height, crop_width = binary_crop.shape[:2]
    if crop_height < 12 or crop_width < 40:
        return 0.0, 0
    vertical = cv2.morphologyEx(
        binary_crop,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(5, int(round(crop_height * 0.55))))
        ),
    )
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tall_lines = 0
    total_line_area = 0
    for contour in contours:
        _, _, line_width, line_height = cv2.boundingRect(contour)
        if line_height >= crop_height * 0.45 and line_width <= crop_width * 0.08:
            tall_lines += 1
            total_line_area += line_width * line_height
    density = total_line_area / float(crop_width * crop_height)
    return density, tall_lines


def morphology_barcode_regions(image: np.ndarray) -> List[ArtifactRegion]:
    """Fallback cho barcode 1D bằng gradient và các vạch dọc lặp lại."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=-1)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=-1)
    gradient = cv2.convertScaleAbs(cv2.subtract(cv2.absdiff(grad_x, 0), cv2.absdiff(grad_y, 0)))
    gradient = cv2.GaussianBlur(gradient, (9, 9), 0)
    _, thresholded = cv2.threshold(
        gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(21, width // 35), max(5, height // 280)),
    )
    connected = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, close_kernel)
    connected = cv2.erode(connected, None, iterations=2)
    connected = cv2.dilate(connected, None, iterations=3)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: List[ArtifactRegion] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < width * 0.08 or box_height < height * 0.015:
            continue
        if box_width > width * 0.90 or box_height > height * 0.30:
            continue
        aspect = box_width / float(max(1, box_height))
        if aspect < 2.0:
            continue
        crop = gray[y : y + box_height, x : x + box_width]
        _, binary_crop = cv2.threshold(
            crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        density, tall_lines = barcode_line_score(binary_crop)
        if tall_lines < 10 or not 0.08 <= density <= 0.75:
            continue
        confidence = min(
            0.94,
            0.68 + min(0.16, tall_lines / 150.0) + min(0.10, density / 2.0),
        )
        regions.append(
            ArtifactRegion(
                kind="barcode",
                box=(int(x), int(y), int(box_width), int(box_height)),
                confidence=round(confidence, 4),
                detector="morphology_vertical_bars",
            )
        )
    return regions


def detect_barcode_regions(image: np.ndarray) -> List[ArtifactRegion]:
    regions = opencv_barcode_regions(image) + morphology_barcode_regions(image)
    return deduplicate_regions(regions, overlap_threshold=0.35)


def detect_logo_regions(image: np.ndarray, max_logos: int = 2) -> List[ArtifactRegion]:
    """Dò logo màu, nhỏ gọn, neo ở mép trên của biểu mẫu bảo hiểm.

    Điều kiện được giữ chặt để không che nhầm tiêu đề đỏ hoặc chữ viết tay.
    Logo đen trắng/ở vị trí bất thường sẽ không bị tự động che và cần cấu hình
    vùng riêng nếu xuất hiện ở bộ tài liệu khác.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = (
        (saturation > 75) & (value > 35) & (value < 250)
    ).astype(np.uint8) * 255

    color_mask[int(height * 0.18) :, :] = 0
    edge_x = max(25, int(round(width * 0.025)))
    edge_y = max(25, int(round(height * 0.015)))
    color_mask[:edge_y, :] = 0
    color_mask[:, :edge_x] = 0
    color_mask[:, width - edge_x :] = 0

    closed = cv2.morphologyEx(
        color_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(15, width // 52), max(7, height // 215)),
        ),
    )
    closed = cv2.dilate(
        closed,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        iterations=1,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[ArtifactRegion] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        width_ratio = box_width / float(width)
        height_ratio = box_height / float(height)
        left_ratio = x / float(width)
        right_ratio = (x + box_width) / float(width)
        anchor_at_edge = left_ratio < 0.12 or right_ratio > 0.88
        if not 0.12 <= width_ratio <= 0.45:
            continue
        if not 0.055 <= height_ratio <= 0.16:
            continue
        if y / float(height) >= 0.14 or not anchor_at_edge:
            continue
        region_mask = color_mask[y : y + box_height, x : x + box_width]
        color_density = cv2.countNonZero(region_mask) / float(box_width * box_height)
        if color_density <= 0.18:
            continue
        confidence = min(
            0.95,
            0.74 + min(0.14, color_density * 0.30) + min(0.07, width_ratio * 0.15),
        )
        candidates.append(
            ArtifactRegion(
                kind="logo",
                box=(int(x), int(y), int(box_width), int(box_height)),
                confidence=round(confidence, 4),
                detector="top_edge_color_logo",
            )
        )
    candidates = deduplicate_regions(candidates, overlap_threshold=0.35)
    return sorted(candidates, key=lambda item: item.confidence, reverse=True)[:max_logos]


def detect_artifact_regions(image: np.ndarray) -> List[ArtifactRegion]:
    qr_regions = detect_qr_regions(image)
    barcode_regions = [
        region
        for region in detect_barcode_regions(image)
        if all(box_iou(region.box, qr.box) < 0.20 for qr in qr_regions)
    ]
    occupied = qr_regions + barcode_regions
    logo_regions = [
        region
        for region in detect_logo_regions(image)
        if all(box_iou(region.box, other.box) < 0.20 for other in occupied)
    ]
    return sorted(
        qr_regions + barcode_regions + logo_regions,
        key=lambda item: (item.box[1], item.box[0], item.kind),
    )


def mask_regions(
    image: np.ndarray,
    regions: Sequence[ArtifactRegion],
    padding: int = 12,
) -> Tuple[np.ndarray, List[Box]]:
    if padding < 0:
        raise ValueError("padding không được âm")
    height, width = image.shape[:2]
    masked = image.copy()
    mask_boxes: List[Box] = []
    for region in regions:
        x, y, box_width, box_height = expand_box(
            region.box, width, height, padding
        )
        masked[y : y + box_height, x : x + box_width] = 255
        mask_boxes.append((x, y, box_width, box_height))
    return masked, mask_boxes


def make_mask_overlay(
    image: np.ndarray,
    regions: Sequence[ArtifactRegion],
    mask_boxes: Sequence[Box],
) -> np.ndarray:
    overlay = image.copy()
    colors = {
        "logo": (255, 120, 0),
        "qr": (0, 180, 0),
        "barcode": (180, 0, 180),
    }
    for region, box in zip(regions, mask_boxes):
        x, y, box_width, box_height = box
        color = colors.get(region.kind, (0, 0, 255))
        cv2.rectangle(
            overlay,
            (x, y),
            (x + box_width, y + box_height),
            color,
            5,
        )
        label_y = y - 10 if y >= 35 else y + 30
        cv2.putText(
            overlay,
            f"{region.kind.upper()} {region.confidence:.2f}",
            (x + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def box_as_json(box: Box) -> Dict[str, int]:
    return {
        "x": int(box[0]),
        "y": int(box[1]),
        "width": int(box[2]),
        "height": int(box[3]),
    }


def mask_panel_artifacts(
    rectified_dir: Path,
    output_dir: Path,
    padding: int = 12,
) -> Dict[str, Any]:
    """Che artifact cho mọi panel trong perspective_manifest.json."""
    rectified_dir = Path(rectified_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = rectified_dir / "perspective_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy perspective manifest: {manifest_path}")
    if output_dir == rectified_dir or output_dir in rectified_dir.parents:
        raise ValueError("Thư mục ảnh masked phải tách khỏi thư mục ảnh chỉnh phối cảnh")
    if padding < 0:
        raise ValueError("padding không được âm")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_specs = source_manifest.get("pages", [])
    if not page_specs:
        raise RuntimeError("Perspective manifest không có trang")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    page_results: List[Dict[str, Any]] = []
    counts = {"logo": 0, "qr": 0, "barcode": 0}
    successful_panels = 0
    needs_review_panels = 0

    for page_spec in page_specs:
        page_number = int(page_spec["page_number"])
        page_dir = output_dir / f"page_{page_number:04d}"
        panel_results: List[Dict[str, Any]] = []
        for panel_spec in page_spec.get("panels", []):
            panel_index = int(panel_spec["panel_index"])
            source_path = rectified_dir / str(panel_spec["rectified_file"])
            image = cv2.imdecode(
                np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"Không thể đọc ảnh panel: {source_path}")
            source_height, source_width = image.shape[:2]
            stem = f"panel_{panel_index:02d}"
            masked_name = f"{stem}_masked.png"
            overlay_name = f"{stem}_mask_overlay.png"
            masked_path = page_dir / masked_name
            overlay_path = page_dir / overlay_name

            try:
                regions = detect_artifact_regions(image)
                masked, mask_boxes = mask_regions(image, regions, padding=padding)
                overlay = make_mask_overlay(image, regions, mask_boxes)
                status = "success"
                error: Optional[str] = None
                successful_panels += 1
            except (cv2.error, ValueError) as detection_error:
                regions = []
                mask_boxes = []
                masked = image.copy()
                overlay = image.copy()
                status = "needs_review"
                error = str(detection_error)
                needs_review_panels += 1
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

            save_png(masked_path, masked)
            save_png(overlay_path, overlay)
            region_entries: List[Dict[str, Any]] = []
            for region, mask_box in zip(regions, mask_boxes):
                counts[region.kind] += 1
                region_entries.append(
                    {
                        "kind": region.kind,
                        "detector": region.detector,
                        "confidence": region.confidence,
                        "detected_bbox": box_as_json(region.box),
                        "masked_bbox": box_as_json(mask_box),
                    }
                )
            panel_results.append(
                {
                    "panel_index": panel_index,
                    "source_file": str(panel_spec["rectified_file"]),
                    "source_width": source_width,
                    "source_height": source_height,
                    "masked_file": str(
                        Path(f"page_{page_number:04d}") / masked_name
                    ).replace("\\", "/"),
                    "overlay_file": str(
                        Path(f"page_{page_number:04d}") / overlay_name
                    ).replace("\\", "/"),
                    "status": status,
                    "error": error,
                    "region_count": len(region_entries),
                    "regions": region_entries,
                    "png_bytes": masked_path.stat().st_size,
                    "png_sha256": sha256_file(masked_path),
                }
            )

        page_results.append(
            {
                "page_number": page_number,
                "status": (
                    "success"
                    if panel_results
                    and all(panel["status"] == "success" for panel in panel_results)
                    else "needs_review"
                ),
                "panel_count": len(panel_results),
                "panels": panel_results,
            }
        )

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "mask_non_text_artifacts",
        "algorithm": "qr_barcode_color_logo_v1",
        "source_perspective_manifest": str(manifest_path),
        "source_perspective_manifest_sha256": sha256_file(manifest_path),
        "mask_fill": "white",
        "mask_padding_pixels": padding,
        "page_count": len(page_results),
        "panel_count": successful_panels + needs_review_panels,
        "successful_panels": successful_panels,
        "needs_review_panels": needs_review_panels,
        "region_counts": counts,
        "region_count": sum(counts.values()),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "pages": page_results,
    }
    atomic_write_json(output_dir / "artifact_mask_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Che logo, QR và barcode khỏi panel; chưa xử lý nền và chưa OCR."
    )
    parser.add_argument("--rectified-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = mask_panel_artifacts(
        args.rectified_dir,
        args.output_dir,
        padding=args.padding,
    )
    counts = result["region_counts"]
    print(
        f"Đã xử lý {result['panel_count']} panel: "
        f"{result['successful_panels']} thành công, "
        f"{result['needs_review_panels']} cần kiểm tra"
    )
    print(
        f"Đã che {result['region_count']} vùng: "
        f"logo={counts['logo']}, QR={counts['qr']}, barcode={counts['barcode']}"
    )
    for page in result["pages"]:
        details = ", ".join(
            f"panel {panel['panel_index']}={panel['region_count']} vùng"
            for panel in page["panels"]
        )
        print(f"Trang {page['page_number']}: {details}")
    print(str(Path(args.output_dir).resolve() / "artifact_mask_manifest.json"))
    return 0 if result["needs_review_panels"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
