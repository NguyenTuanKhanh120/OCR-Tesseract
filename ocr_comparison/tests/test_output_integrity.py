"""
Unit test kiểm tra bảo đảm tính bất biến (hash không thay đổi) của thư mục output/
trước và sau khi chạy quá trình so sánh (mục 18 CONTEXT_COMPARE_APPLE_VISION_TESSERACT.md).
"""

import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from ocr_comparison.config import ComparisonConfig
from ocr_comparison.compare_outputs import run_comparison


def get_tree_hashes(directory: Path) -> dict[str, str]:
    """Lấy danh sách mã hash SHA256 của toàn bộ các file trong thư mục."""
    hashes = {}
    for f in sorted(directory.rglob("*")):
        if f.is_file():
            hasher = hashlib.sha256()
            with open(f, "rb") as fp:
                while chunk := fp.read(8192):
                    hasher.update(chunk)
            rel = str(f.relative_to(directory))
            hashes[rel] = hasher.hexdigest()
    return hashes


class TestOutputIntegrity(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(tempfile.mkdtemp(prefix="ocr_integrity_"))
        self.apple_dir = self.test_root / "apple_output"
        self.tess_dir = self.test_root / "tesseract_output"
        self.report_dir = self.test_root / "comparison_results"

        self.apple_dir.mkdir()
        self.tess_dir.mkdir()
        self.report_dir.mkdir()

        # Tạo file mẫu trong apple_output
        sub = self.apple_dir / "CATEGORY_A"
        sub.mkdir()
        (sub / "doc1_ocr.txt").write_text("--- [TRANG 1/1] ---\nVăn bản Apple Vision", encoding="utf-8")

        # Tạo file mẫu trong tesseract_output
        sub_tess = self.tess_dir / "CATEGORY_A"
        sub_tess.mkdir()
        (sub_tess / "doc1_ocr.txt").write_text("--- [TRANG 1/1] ---\nVăn bản Tesseract", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_apple_dir_unmodified_after_comparison(self):
        # 1. Đo hash trước khi chạy so sánh
        before_hashes = get_tree_hashes(self.apple_dir)
        self.assertTrue(len(before_hashes) > 0)

        # 2. Chạy quá trình so sánh
        config = ComparisonConfig(
            apple_dir=self.apple_dir,
            tesseract_dir=self.tess_dir,
            report_dir=self.report_dir,
            run_id="test_run",
        )
        report_path = run_comparison(config)

        # 3. Đo hash sau khi chạy so sánh
        after_hashes = get_tree_hashes(self.apple_dir)

        # 4. Kiểm tra tuyệt đối không có sự biến đổi
        self.assertEqual(before_hashes, after_hashes, "LỖI AN TOÀN: Thư mục Apple Vision đã bị thay đổi!")

        # Báo cáo phải được tạo trong report_dir
        self.assertTrue((report_path / "summary.json").exists())
        self.assertTrue((report_path / "page_pairwise_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
