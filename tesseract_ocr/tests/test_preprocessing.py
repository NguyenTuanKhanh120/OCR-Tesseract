"""
Unit test cho module tiền xử lý ảnh (OpenCV + NumPy).
"""

import unittest
import numpy as np
import cv2

from tesseract_ocr.preprocessing import (
    crop_black_borders,
    detect_deskew_angle,
    enhance_contrast_if_needed,
    preprocess_image,
    remove_long_colored_lines,
    rotate_image_expand,
)


class TestPreprocessing(unittest.TestCase):
    def test_rotate_image_expand(self):
        # Tạo ảnh trắng 100x200
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        # Xoay 90 độ -> kích thước mới phải xấp xỉ 200x100
        rot = rotate_image_expand(img, 90)
        h, w = rot.shape[:2]
        self.assertEqual((h, w), (200, 100))

        # Xoay 45 độ -> kích thước canvas phải lớn hơn để không bị cắt xén
        rot45 = rotate_image_expand(img, 45)
        h45, w45 = rot45.shape[:2]
        self.assertGreater(h45, 100)
        self.assertGreater(w45, 200)

    def test_crop_black_borders(self):
        # Ảnh 200x200 có viền đen 20 pixel bao quanh nội dung trắng
        img = np.zeros((200, 200), dtype=np.uint8)
        img[30:170, 30:170] = 255  # Vùng trắng ở giữa
        cropped = crop_black_borders(img)
        # Kích thước sau khi cắt phải nhỏ hơn 200x200
        self.assertLess(cropped.shape[0], 200)
        self.assertLess(cropped.shape[1], 200)

    def test_enhance_contrast(self):
        # Ảnh có độ tương phản rất thấp (toàn bộ giá trị từ 120 đến 130)
        low_contrast = np.random.randint(120, 130, size=(100, 100), dtype=np.uint8)
        enhanced, applied = enhance_contrast_if_needed(low_contrast)
        self.assertTrue(applied)
        self.assertGreater(np.std(enhanced), np.std(low_contrast))

    def test_preprocess_mode_none(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        out, meta = preprocess_image(img, mode="none")
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(meta.orientation_rotation, 0)
        self.assertEqual(meta.deskew_angle, 0.0)

    def test_preprocess_mode_auto(self):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Vẽ vài dòng chữ đen
        cv2.putText(img, "TEST OCR", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        out, meta = preprocess_image(img, mode="auto")
        # Kết quả auto trả về ảnh xám (2 chiều)
        self.assertEqual(len(out.shape), 2)
        self.assertIsInstance(meta.processing_seconds, float)

    def test_preprocess_mode_grayscale_does_not_binarize(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :25] = (100, 100, 100)
        img[:, 25:] = (200, 200, 200)
        out, meta = preprocess_image(img, mode="grayscale")
        self.assertEqual(len(out.shape), 2)
        self.assertGreater(len(np.unique(out)), 1)
        self.assertFalse(meta.threshold_applied)
        self.assertIn("grayscale_document_cleanup", meta.notes)

    def test_remove_long_colored_line_preserves_local_red_region(self):
        image = np.full((220, 500, 3), 255, dtype=np.uint8)
        image[100:104, :] = (255, 255, 0)  # cyan theo thứ tự BGR
        image[170:200, 220:280] = (0, 0, 255)  # vùng đỏ cục bộ như con dấu

        cleaned, line_count = remove_long_colored_lines(image)

        self.assertEqual(line_count, 1)
        self.assertGreater(int(cleaned[101, 250].min()), 240)
        self.assertTrue(np.array_equal(cleaned[180, 250], np.array([0, 0, 255])))

    def test_does_not_remove_wide_colored_text(self):
        image = np.full((300, 900, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "TIEU DE MAU XANH CUA TAI LIEU",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 180, 0),
            3,
        )

        cleaned, line_count = remove_long_colored_lines(image)

        self.assertEqual(line_count, 0)
        self.assertTrue(np.array_equal(cleaned, image))

    def test_preprocess_grayscale_removes_long_colored_line(self):
        image = np.full((220, 500, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            "DIEU 5: NOI DUNG VAN BAN",
            (35, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        image[100:104, :] = (255, 255, 0)

        output, meta = preprocess_image(image, mode="grayscale")

        self.assertEqual(len(output.shape), 2)
        self.assertIn("colored_lines_removed=1", meta.notes)
        self.assertGreater(float(np.mean(output[100:104, :])), 230.0)

    def test_deskew_reduces_synthetic_text_angle(self):
        image = np.full((500, 900), 255, dtype=np.uint8)
        for row, text in enumerate(
            [
                "CONG HOA XA HOI CHU NGHIA VIET NAM",
                "NOI QUY PHONG CHAY VA CHUA CHAY",
                "DIEU MOT NOI DUNG VAN BAN THU NGHIEM",
                "DIEU HAI NOI DUNG VAN BAN THU NGHIEM",
                "DIEU BA NOI DUNG VAN BAN THU NGHIEM",
            ]
        ):
            cv2.putText(
                image,
                text,
                (45, 90 + row * 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                0,
                2,
            )
        tilted = rotate_image_expand(image, 4.0)
        detected = detect_deskew_angle(tilted)
        corrected = rotate_image_expand(tilted, detected)
        residual = detect_deskew_angle(corrected)

        self.assertGreater(abs(detected), 2.0)
        self.assertLess(abs(residual), 1.0)


if __name__ == "__main__":
    unittest.main()
