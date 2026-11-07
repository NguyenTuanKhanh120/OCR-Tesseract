"""Xử lý nền màu và tăng tương phản panel trước OCR.

Đây chỉ là bước 5 của pipeline OCR theo vùng. Module tự nhận biết nền vàng
để chọn kênh xanh lá, dùng ảnh xám cho tài liệu không vàng, áp dụng CLAHE và
tạo thêm bản nhị phân Sauvola. Mọi thông số được ghi vào manifest.

Không chọn PSM, không gọi Tesseract và không ghép kết quả văn bản.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from tesseract_ocr.document_renderer import atomic_write_json, save_png, sha256_file


if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def estimate_yellow_fraction(image: np.ndarray) -> float:
    """Ước lượng tỷ lệ pixel nền vàng trong không gian HSV của OpenCV."""
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")
    if image.ndim != 3 or image.shape[2] != 3:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    yellow = (
        (hue >= 15)
        & (hue <= 45)
        & (saturation >= 35)
        & (value >= 80)
    )
    return float(np.mean(yellow))


def select_document_channel(
    image: np.ndarray,
    mode: str = "auto",
    yellow_threshold: float = 0.08,
) -> Tuple[np.ndarray, str, float]:
    """Chọn kênh xanh cho nền vàng, ảnh xám cho các trường hợp còn lại."""
    if mode not in ("auto", "green", "grayscale"):
        raise ValueError("mode chỉ chấp nhận auto, green hoặc grayscale")
    if not 0.0 <= yellow_threshold <= 1.0:
        raise ValueError("yellow_threshold phải nằm trong khoảng 0..1")
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào rỗng")

    yellow_fraction = estimate_yellow_fraction(image)
    if image.ndim == 2:
        return image.copy(), "grayscale", yellow_fraction

    selected_mode = mode
    if mode == "auto":
        selected_mode = (
            "green" if yellow_fraction >= yellow_threshold else "grayscale"
        )
    if selected_mode == "green":
        channel = image[:, :, 1].copy()
    else:
        channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return channel, selected_mode, yellow_fraction


def apply_clahe(
    channel: np.ndarray,
    clip_limit: float = 2.2,
    tile_grid_size: int = 8,
) -> np.ndarray:
    if channel.ndim != 2:
        raise ValueError("CLAHE yêu cầu ảnh một kênh")
    if clip_limit <= 0:
        raise ValueError("clip_limit phải lớn hơn 0")
    if tile_grid_size <= 0:
        raise ValueError("tile_grid_size phải lớn hơn 0")
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(channel)


def sauvola_binarize(
    image: np.ndarray,
    window_size: int = 41,
    k: float = 0.20,
    dynamic_range: float = 128.0,
) -> np.ndarray:
    """Nhị phân Sauvola bằng box filter, giữ tốt nét mảnh trên nền không đều."""
    if image.ndim != 2:
        raise ValueError("Sauvola yêu cầu ảnh một kênh")
    if window_size < 3:
        raise ValueError("window_size phải từ 3 trở lên")
    if window_size % 2 == 0:
        window_size += 1
    if dynamic_range <= 0:
        raise ValueError("dynamic_range phải lớn hơn 0")

    source = image.astype(np.float32)
    mean = cv2.boxFilter(
        source,
        cv2.CV_32F,
        (window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    square_mean = cv2.boxFilter(
        source * source,
        cv2.CV_32F,
        (window_size, window_size),
        normalize=True,
        borderType=cv2.BORDER_REPLICATE,
    )
    variance = np.maximum(square_mean - (mean * mean), 0.0)
    standard_deviation = np.sqrt(variance)
    threshold = mean * (
        1.0 + float(k) * ((standard_deviation / float(dynamic_range)) - 1.0)
    )
    return np.where(source > threshold, 255, 0).astype(np.uint8)


def foreground_ratio(binary: np.ndarray) -> float:
    if binary.ndim != 2:
        raise ValueError("Ảnh nhị phân phải có một kênh")
    return float(np.mean(binary < 128))


def preprocess_background(
    image: np.ndarray,
    channel_mode: str = "auto",
    yellow_threshold: float = 0.08,
    clahe_clip_limit: float = 2.2,
    clahe_tile_size: int = 8,
    sauvola_window: int = 41,
    sauvola_k: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    channel, selected_channel, yellow_fraction = select_document_channel(
        image,
        mode=channel_mode,
        yellow_threshold=yellow_threshold,
    )
    enhanced = apply_clahe(
        channel,
        clip_limit=clahe_clip_limit,
        tile_grid_size=clahe_tile_size,
    )
    binary = sauvola_binarize(
        enhanced,
        window_size=sauvola_window,
        k=sauvola_k,
    )
    input_std = float(np.std(channel))
    enhanced_std = float(np.std(enhanced))
    ratio = foreground_ratio(binary)
    metrics: Dict[str, Any] = {
        "selected_channel": selected_channel,
        "yellow_fraction": round(yellow_fraction, 6),
        "yellow_background_detected": bool(yellow_fraction >= yellow_threshold),
        "input_mean": round(float(np.mean(channel)), 3),
        "input_std": round(input_std, 3),
        "enhanced_mean": round(float(np.mean(enhanced)), 3),
        "enhanced_std": round(enhanced_std, 3),
        "contrast_gain": round(enhanced_std / max(input_std, 1e-6), 4),
        "binary_foreground_ratio": round(ratio, 6),
    }
    return enhanced, binary, metrics


def process_masked_panels(
    masked_dir: Path,
    output_dir: Path,
    channel_mode: str = "auto",
    yellow_threshold: float = 0.08,
    clahe_clip_limit: float = 2.2,
    clahe_tile_size: int = 8,
    sauvola_window: int = 41,
    sauvola_k: float = 0.20,
) -> Dict[str, Any]:
    """Xử lý nền mọi panel trong artifact_mask_manifest.json."""
    masked_dir = Path(masked_dir).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = masked_dir / "artifact_mask_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy artifact mask manifest: {manifest_path}")
    if output_dir == masked_dir or output_dir in masked_dir.parents:
        raise ValueError("Thư mục tiền xử lý phải tách khỏi thư mục ảnh masked")
    if channel_mode not in ("auto", "green", "grayscale"):
        raise ValueError("channel_mode chỉ chấp nhận auto, green hoặc grayscale")
    if not 0.0 <= yellow_threshold <= 1.0:
        raise ValueError("yellow_threshold phải nằm trong khoảng 0..1")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    page_specs = source_manifest.get("pages", [])
    if not page_specs:
        raise RuntimeError("Artifact mask manifest không có trang")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    page_results: List[Dict[str, Any]] = []
    successful_panels = 0
    needs_review_panels = 0
    channel_counts = {"green": 0, "grayscale": 0}

    for page_spec in page_specs:
        page_number = int(page_spec["page_number"])
        page_dir = output_dir / f"page_{page_number:04d}"
        panel_results: List[Dict[str, Any]] = []
        for panel_spec in page_spec.get("panels", []):
            panel_index = int(panel_spec["panel_index"])
            source_path = masked_dir / str(panel_spec["masked_file"])
            image = cv2.imdecode(
                np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"Không thể đọc ảnh panel: {source_path}")
            source_height, source_width = image.shape[:2]
            stem = f"panel_{panel_index:02d}"
            enhanced_name = f"{stem}_enhanced.png"
            binary_name = f"{stem}_binary.png"
            enhanced_path = page_dir / enhanced_name
            binary_path = page_dir / binary_name

            enhanced, binary, metrics = preprocess_background(
                image,
                channel_mode=channel_mode,
                yellow_threshold=yellow_threshold,
                clahe_clip_limit=clahe_clip_limit,
                clahe_tile_size=clahe_tile_size,
                sauvola_window=sauvola_window,
                sauvola_k=sauvola_k,
            )
            ratio = float(metrics["binary_foreground_ratio"])
            status = "success" if 0.003 <= ratio <= 0.60 else "needs_review"
            review_reason = (
                None
                if status == "success"
                else f"Tỷ lệ foreground bất thường: {ratio:.4f}"
            )
            save_png(enhanced_path, enhanced)
            save_png(binary_path, binary)

            selected_channel = str(metrics["selected_channel"])
            channel_counts[selected_channel] += 1
            if status == "success":
                successful_panels += 1
            else:
                needs_review_panels += 1
            panel_results.append(
                {
                    "panel_index": panel_index,
                    "source_file": str(panel_spec["masked_file"]),
                    "source_width": source_width,
                    "source_height": source_height,
                    "enhanced_file": str(
                        Path(f"page_{page_number:04d}") / enhanced_name
                    ).replace("\\", "/"),
                    "binary_file": str(
                        Path(f"page_{page_number:04d}") / binary_name
                    ).replace("\\", "/"),
                    "status": status,
                    "review_reason": review_reason,
                    "metrics": metrics,
                    "enhanced_png_bytes": enhanced_path.stat().st_size,
                    "enhanced_png_sha256": sha256_file(enhanced_path),
                    "binary_png_bytes": binary_path.stat().st_size,
                    "binary_png_sha256": sha256_file(binary_path),
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
        "operation": "enhance_colored_background",
        "algorithm": "auto_green_clahe_sauvola_v1",
        "source_artifact_mask_manifest": str(manifest_path),
        "source_artifact_mask_manifest_sha256": sha256_file(manifest_path),
        "channel_mode": channel_mode,
        "yellow_threshold": yellow_threshold,
        "clahe": {
            "clip_limit": clahe_clip_limit,
            "tile_grid_size": clahe_tile_size,
        },
        "sauvola": {
            "window_size": sauvola_window if sauvola_window % 2 else sauvola_window + 1,
            "k": sauvola_k,
            "dynamic_range": 128.0,
        },
        "page_count": len(page_results),
        "panel_count": successful_panels + needs_review_panels,
        "successful_panels": successful_panels,
        "needs_review_panels": needs_review_panels,
        "selected_channel_counts": channel_counts,
        "processing_seconds": round(time.perf_counter() - started, 3),
        "pages": page_results,
    }
    atomic_write_json(output_dir / "background_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Xử lý nền bằng kênh xanh/ảnh xám, CLAHE và Sauvola; chưa OCR."
    )
    parser.add_argument("--masked-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--channel", choices=["auto", "green", "grayscale"], default="auto"
    )
    parser.add_argument("--yellow-threshold", type=float, default=0.08)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.2)
    parser.add_argument("--clahe-tile-size", type=int, default=8)
    parser.add_argument("--sauvola-window", type=int, default=41)
    parser.add_argument("--sauvola-k", type=float, default=0.20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = process_masked_panels(
        args.masked_dir,
        args.output_dir,
        channel_mode=args.channel,
        yellow_threshold=args.yellow_threshold,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=args.clahe_tile_size,
        sauvola_window=args.sauvola_window,
        sauvola_k=args.sauvola_k,
    )
    print(
        f"Đã xử lý {result['panel_count']} panel: "
        f"{result['successful_panels']} thành công, "
        f"{result['needs_review_panels']} cần kiểm tra"
    )
    counts = result["selected_channel_counts"]
    print(f"Kênh đã chọn: green={counts['green']}, grayscale={counts['grayscale']}")
    for page in result["pages"]:
        details = ", ".join(
            f"panel {panel['panel_index']}={panel['metrics']['selected_channel']} "
            f"(yellow {panel['metrics']['yellow_fraction']:.3f}, "
            f"foreground {panel['metrics']['binary_foreground_ratio']:.3f})"
            for panel in page["panels"]
        )
        print(f"Trang {page['page_number']}: {details}")
    print(str(Path(args.output_dir).resolve() / "background_manifest.json"))
    return 0 if result["needs_review_panels"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
