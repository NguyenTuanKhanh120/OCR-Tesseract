"""Phát hiện và chia các panel lớn từ ảnh trang PDF đã render.

Đây chỉ là bước 2 của pipeline OCR theo vùng. Module dùng các đường viền dài
để tìm panel, cắt theo bounding box trục thẳng và tạo ảnh overlay kiểm tra.
Không chỉnh phối cảnh, không tiền xử lý OCR và không chạy Tesseract.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


Box = Tuple[int, int, int, int]


def box_area(box: Box) -> int:
    return max(0, box[2]) * max(0, box[3])


def intersection_over_smaller(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    smaller = min(box_area(box_a), box_area(box_b))
    return intersection / smaller if smaller else 0.0


def deduplicate_boxes(boxes: Iterable[Box], overlap_threshold: float = 0.85) -> List[Box]:
    """Loại viền trong/ngoài gần trùng nhau, giữ bounding box lớn hơn."""
    selected: List[Box] = []
    for box in sorted(boxes, key=box_area, reverse=True):
        if any(intersection_over_smaller(box, existing) >= overlap_threshold for existing in selected):
            continue
        selected.append(box)
    return selected


def select_non_overlapping_panels(
    boxes: Sequence[Box],
    expected_panels: Optional[int] = None,
    max_panels: int = 3,
) -> List[Box]:
    """Chọn nhóm panel lớn, không lồng nhau, có mép trên/dưới tương đối thẳng hàng."""
    candidates = list(boxes)
    if not candidates:
        return []

    target_sizes = [expected_panels] if expected_panels else list(range(2, max_panels + 1))
    best_group: Optional[Tuple[float, Tuple[Box, ...]]] = None

    for target_size in target_sizes:
        if target_size is None or target_size <= 0 or len(candidates) < target_size:
            continue
        for group in itertools.combinations(candidates, target_size):
            if any(
                intersection_over_smaller(left, right) > 0.20
                for left, right in itertools.combinations(group, 2)
            ):
                continue

            sorted_group = tuple(sorted(group, key=lambda box: box[0]))
            heights = [box[3] for box in sorted_group]
            tops = [box[1] for box in sorted_group]
            bottoms = [box[1] + box[3] for box in sorted_group]
            median_height = float(np.median(heights))
            if median_height <= 0:
                continue
            top_spread = (max(tops) - min(tops)) / median_height
            bottom_spread = (max(bottoms) - min(bottoms)) / median_height
            height_spread = (max(heights) - min(heights)) / median_height
            alignment_penalty = top_spread + bottom_spread + height_spread
            total_area = float(sum(box_area(box) for box in sorted_group))
            # Khi tự động, thưởng nhẹ cho ba panel nếu chúng thực sự thẳng hàng.
            auto_panel_bonus = 1.03 if expected_panels is None and target_size == 3 else 1.0
            score = total_area * auto_panel_bonus / (1.0 + 2.5 * alignment_penalty)
            if best_group is None or score > best_group[0]:
                best_group = (score, sorted_group)

    if best_group is not None:
        return list(best_group[1])

    # Fallback để tạo artifact kiểm tra thay vì im lặng bỏ trang.
    return sorted(candidates, key=box_area, reverse=True)[: max(1, expected_panels or max_panels)]


def detect_panel_boxes(
    image: np.ndarray,
    expected_panels: Optional[int] = None,
    max_panels: int = 3,
) -> List[Box]:
    """Tìm 2-3 panel lớn dựa trên đường viền dài ngang/dọc."""
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    height, width = gray.shape[:2]
    if height < 100 or width < 100:
        raise ValueError(f"Ảnh quá nhỏ để phát hiện panel: {width}x{height}")

    # Ngưỡng đủ rộng để bắt đường viền xanh/đen trên nền trắng hoặc nền màu nhạt.
    _, inverted = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(40, width // 12), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(60, height // 12))
    )
    horizontal = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, vertical_kernel)
    line_mask = cv2.bitwise_or(horizontal, vertical)
    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )

    contours, _ = cv2.findContours(line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[Box] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        width_ratio = box_width / width
        height_ratio = box_height / height
        if not 0.18 <= width_ratio <= 0.60:
            continue
        if not 0.25 <= height_ratio <= 0.95:
            continue
        if y > height * 0.60:
            continue
        candidates.append((int(x), int(y), int(box_width), int(box_height)))

    unique = deduplicate_boxes(candidates)
    return select_non_overlapping_panels(
        unique,
        expected_panels=expected_panels,
        max_panels=max_panels,
    )


def padded_box(box: Box, image_width: int, image_height: int, padding: int) -> Box:
    x, y, width, height = box
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return left, top, right - left, bottom - top


def split_rendered_pages(
    render_dir: Path,
    output_dir: Path,
    expected_panels: Optional[int] = None,
    padding: int = 20,
) -> Dict[str, Any]:
    """Chia panel cho toàn bộ ảnh trang được mô tả bởi render_manifest.json."""
    render_dir = Path(render_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = render_dir / "render_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy render manifest: {manifest_path}")
    if output_dir == render_dir or output_dir in render_dir.parents:
        raise ValueError("Thư mục panel phải tách khỏi thư mục ảnh render")
    if expected_panels is not None and expected_panels not in (2, 3):
        raise ValueError("expected_panels chỉ chấp nhận 2, 3 hoặc bỏ trống để tự động")
    if padding < 0:
        raise ValueError("padding không được âm")

    render_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_specs = render_manifest.get("pages", [])
    if not page_specs:
        raise RuntimeError("Render manifest không có trang")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    page_results: List[Dict[str, Any]] = []

    for page_spec in page_specs:
        page_number = int(page_spec["page_number"])
        source_image_path = render_dir / str(page_spec["file"])
        image = cv2.imdecode(
            np.fromfile(str(source_image_path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise RuntimeError(f"Không thể đọc ảnh trang: {source_image_path}")
        image_height, image_width = image.shape[:2]
        boxes = detect_panel_boxes(image, expected_panels=expected_panels)
        status = (
            "success"
            if boxes and (expected_panels is None or len(boxes) == expected_panels)
            else "needs_review"
        )

        page_dir = output_dir / f"page_{page_number:04d}"
        overlay = image.copy()
        panels: List[Dict[str, Any]] = []
        for panel_index, box in enumerate(sorted(boxes, key=lambda value: value[0]), 1):
            crop_box = padded_box(box, image_width, image_height, padding)
            x, y, width, height = crop_box
            crop = image[y : y + height, x : x + width].copy()
            panel_name = f"panel_{panel_index:02d}.png"
            panel_path = page_dir / panel_name
            save_png(panel_path, crop)

            color = (0, 180, 0)
            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 8)
            cv2.putText(
                overlay,
                f"PANEL {panel_index}",
                (x + 20, max(40, y + 55)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.6,
                color,
                4,
                cv2.LINE_AA,
            )
            panels.append(
                {
                    "panel_index": panel_index,
                    "file": str(Path(f"page_{page_number:04d}") / panel_name).replace("\\", "/"),
                    "bbox": {"x": x, "y": y, "width": width, "height": height},
                    "normalized_bbox": {
                        "x": round(x / image_width, 6),
                        "y": round(y / image_height, 6),
                        "width": round(width / image_width, 6),
                        "height": round(height / image_height, 6),
                    },
                    "png_bytes": panel_path.stat().st_size,
                    "png_sha256": sha256_file(panel_path),
                }
            )

        overlay_name = f"page_{page_number:04d}_panels_overlay.png"
        overlay_path = page_dir / overlay_name
        save_png(overlay_path, overlay)
        page_results.append(
            {
                "page_number": page_number,
                "source_image": str(page_spec["file"]),
                "source_width": image_width,
                "source_height": image_height,
                "expected_panels": expected_panels,
                "detected_panels": len(panels),
                "status": status,
                "overlay_file": str(Path(f"page_{page_number:04d}") / overlay_name).replace("\\", "/"),
                "panels": panels,
            }
        )

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "operation": "split_page_panels",
        "algorithm": "long_border_panels_v1",
        "source_render_manifest": str(manifest_path),
        "source_render_manifest_sha256": sha256_file(manifest_path),
        "expected_panels_per_page": expected_panels,
        "padding_pixels": padding,
        "page_count": len(page_results),
        "successful_pages": sum(page["status"] == "success" for page in page_results),
        "needs_review_pages": sum(page["status"] != "success" for page in page_results),
        "processing_seconds": round(time.perf_counter() - started, 3),
        "pages": page_results,
    }
    atomic_write_json(output_dir / "panels_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chia các panel lớn từ ảnh PDF đã render; chưa OCR."
    )
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-panels", type=int, choices=[2, 3], default=None)
    parser.add_argument("--padding", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = split_rendered_pages(
        args.render_dir,
        args.output_dir,
        expected_panels=args.expected_panels,
        padding=args.padding,
    )
    print(
        f"Đã xử lý {result['page_count']} trang: "
        f"{result['successful_pages']} thành công, "
        f"{result['needs_review_pages']} cần kiểm tra"
    )
    for page in result["pages"]:
        print(
            f"Trang {page['page_number']}: {page['detected_panels']} panel "
            f"({page['status']})"
        )
    print(str(Path(args.output_dir).resolve() / "panels_manifest.json"))
    return 0 if result["needs_review_pages"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
