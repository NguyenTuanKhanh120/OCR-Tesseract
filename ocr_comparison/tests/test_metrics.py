"""
Unit test kiểm tra tính chính xác của các thuật toán tính Metric OCR.
"""

import unittest

from ocr_comparison.metrics import (
    compute_cer,
    compute_diacritic_error_rate,
    compute_multiset_word_f1,
    compute_pattern_accuracy,
    compute_similarity,
    compute_wer,
    evaluate_page_pair,
    levenshtein_distance,
    normalize_text,
    REGEX_DATE,
    REGEX_DIGITS,
)


class TestMetrics(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance(list("kitten"), list("sitting")), 3)
        self.assertEqual(levenshtein_distance(list(""), list("abc")), 3)
        self.assertEqual(levenshtein_distance(list("abc"), list("abc")), 0)

    def test_cer_and_wer_perfect(self):
        ref = "Cộng hòa Xã hội Chủ nghĩa Việt Nam"
        hyp = "Cộng hòa Xã hội Chủ nghĩa Việt Nam"
        self.assertEqual(compute_cer(ref, hyp), 0.0)
        self.assertEqual(compute_wer(ref, hyp), 0.0)
        p, r, f1 = compute_multiset_word_f1(ref, hyp)
        self.assertEqual(f1, 1.0)

    def test_cer_vietnamese_diacritics_preserved(self):
        # Mất dấu tiếng Việt phải được tính là lỗi ký tự trong CER chính
        ref = "Hà Nội"
        hyp = "Ha Noi"
        # 'à' -> 'a', 'ộ' -> 'o' (2 lỗi thay thế trên 6 ký tự)
        cer = compute_cer(ref, hyp)
        self.assertGreater(cer, 0.0)
        self.assertAlmostEqual(cer, 2 / 6, places=4)

    def test_multiset_word_f1_duplicate_tokens(self):
        # Kiểm tra tính multiset: đếm số lần xuất hiện chứ không dùng set()
        ref = "tiền tiền tiền"
        hyp = "tiền"
        # hyp chỉ có 1 từ 'tiền', trong khi ref có 3
        # overlap = 1 -> precision = 1/1 = 1.0, recall = 1/3 = 0.333
        p, r, f1 = compute_multiset_word_f1(ref, hyp)
        self.assertEqual(p, 1.0)
        self.assertAlmostEqual(r, 1 / 3, places=3)
        self.assertLess(f1, 1.0)

    def test_pattern_accuracy(self):
        ref = "Hợp đồng số 12345 ký ngày 15/08/2023 giá trị 500.000 VND"
        hyp = "Hợp đồng số 12345 ký ngày 15/08/2023 giá trị 400.000 VND"

        digit_acc = compute_pattern_accuracy(ref, hyp, REGEX_DIGITS)
        # Các số trong ref: '12345', '15', '08', '2023', '500.000'
        self.assertIsNotNone(digit_acc)
        self.assertGreater(digit_acc, 0.5)

        date_acc = compute_pattern_accuracy(ref, hyp, REGEX_DATE)
        self.assertEqual(date_acc, 1.0)

    def test_evaluate_page_pair_without_ground_truth(self):
        # Chế độ KHÔNG có ground truth: không kết luận winner, không có CER/WER
        metric = evaluate_page_pair(
            doc_id="D1",
            relative_path="PCCC/d1",
            category="PCCC",
            page_number=1,
            apple_text="Công an TP Hà Nội",
            tesseract_text="Công an TP Ha Noi",
            apple_status="success",
            tesseract_status="success",
            ground_truth_text=None,
        )
        self.assertFalse(metric.ground_truth_available)
        self.assertIsNone(metric.winner)
        self.assertIsNone(metric.apple_cer_pct)
        self.assertIsNone(metric.tesseract_cer_pct)
        self.assertGreater(metric.pairwise_similarity_pct, 50.0)

    def test_evaluate_page_pair_with_ground_truth(self):
        # Chế độ CÓ ground truth: có winner và CER/WER
        gt = "Công an Thành phố Hà Nội"
        apple_text = "Công an Thành phố Hà Nội"  # Hoàn hảo
        tess_text = "Cong an Thanh pho Ha Noi"  # Mất dấu

        metric = evaluate_page_pair(
            doc_id="D1",
            relative_path="PCCC/d1",
            category="PCCC",
            page_number=1,
            apple_text=apple_text,
            tesseract_text=tess_text,
            apple_status="success",
            tesseract_status="success",
            ground_truth_text=gt,
            cer_threshold=0.05,
        )
        self.assertTrue(metric.ground_truth_available)
        self.assertEqual(metric.winner, "apple")
        self.assertEqual(metric.apple_cer_pct, 0.0)
        self.assertGreater(metric.tesseract_cer_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
