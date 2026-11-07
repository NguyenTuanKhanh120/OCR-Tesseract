"""
Tính toán các chỉ số đánh giá OCR (CER, WER, F1 multiset, Business metrics, Pairwise difference).
Tuân thủ đầy đủ quy định tại mục 10 và 12 của tài liệu thiết kế.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Bảng nguyên âm tiếng Việt có dấu
VIETNAMESE_VOWELS_ACCENTED = set(
    "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ"
    "ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ"
)

# Regex cho các thực thể nghiệp vụ
REGEX_DIGITS = re.compile(r"\b\d+(?:[.,]\d+)*\b")
REGEX_DATE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
REGEX_MONEY = re.compile(r"\b\d{1,3}(?:[.,]\d{3})+(?:\s*(?:VND|đ|dong|USD|\$))?\b", re.IGNORECASE)
REGEX_CODE = re.compile(r"\b[A-Z0-9]{2,}(?:[-/][A-Z0-9]+)+\b")


def normalize_text(text: str, mode: str = "normalized") -> str:
    """
    Chuẩn hóa chuỗi văn bản:
    - strict: Unicode NFC, giữ nguyên chữ hoa/thường và dấu câu.
    - normalized: Unicode NFC, chuẩn hóa khoảng trắng/xuống dòng, không bỏ dấu tiếng Việt.
    """
    if not text:
        return ""

    # Luôn chuẩn hóa Unicode NFC
    norm = unicodedata.normalize("NFC", text)

    if mode == "strict":
        return norm

    # normalized: thu gọn nhiều khoảng trắng và xuống dòng thành một dấu cách
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def levenshtein_distance(seq1: List[Any], seq2: List[Any]) -> int:
    """
    Tính khoảng cách Levenshtein giữa hai chuỗi (ký tự hoặc token).
    Triển khai thuật toán 2 dòng tối ưu O(min(M, N)) bộ nhớ.
    """
    if seq1 == seq2:
        return 0
    if not seq1:
        return len(seq2)
    if not seq2:
        return len(seq1)

    # Đảm bảo seq1 ngắn hơn seq2 để tối ưu bộ nhớ
    if len(seq1) > len(seq2):
        seq1, seq2 = seq2, seq1

    previous_row = list(range(len(seq1) + 1))
    for j, c2 in enumerate(seq2):
        current_row = [j + 1] * (len(seq1) + 1)
        for i, c1 in enumerate(seq1):
            insertions = previous_row[i + 1] + 1
            deletions = current_row[i] + 1
            substitutions = previous_row[i] + (0 if c1 == c2 else 1)
            current_row[i + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def compute_cer(reference: str, hypothesis: str, mode: str = "normalized") -> float:
    """
    Tính Character Error Rate (CER).
    CER = Levenshtein(ref_chars, hyp_chars) / len(ref_chars).
    """
    ref_norm = normalize_text(reference, mode)
    hyp_norm = normalize_text(hypothesis, mode)

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_wer(reference: str, hypothesis: str, mode: str = "normalized") -> float:
    """
    Tính Word Error Rate (WER).
    WER = Levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens).
    """
    ref_norm = normalize_text(reference, mode)
    hyp_norm = normalize_text(hypothesis, mode)

    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    dist = levenshtein_distance(ref_tokens, hyp_tokens)
    return dist / len(ref_tokens)


def compute_multiset_word_f1(
    reference: str,
    hypothesis: str,
    mode: str = "normalized",
) -> Tuple[float, float, float]:
    """
    Tính Word Precision, Recall và F1 có tính số lần xuất hiện (Multiset Counter, không dùng set).
    """
    ref_norm = normalize_text(reference, mode)
    hyp_norm = normalize_text(hypothesis, mode)

    ref_tokens = ref_norm.split()
    hyp_tokens = hyp_norm.split()

    if not ref_tokens and not hyp_tokens:
        return 1.0, 1.0, 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0, 0.0, 0.0

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)

    # Giao của hai đa tập hợp (min số lần xuất hiện của mỗi từ)
    overlap_count = 0
    for word, h_count in hyp_counts.items():
        if word in ref_counts:
            overlap_count += min(h_count, ref_counts[word])

    precision = overlap_count / len(hyp_tokens) if hyp_tokens else 0.0
    recall = overlap_count / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_similarity(text1: str, text2: str) -> float:
    """
    Tính độ tương đồng văn bản (0..100%).
    Dựa trên Levenshtein character distance.
    """
    t1_norm = normalize_text(text1, "normalized")
    t2_norm = normalize_text(text2, "normalized")

    if not t1_norm and not t2_norm:
        return 100.0

    max_len = max(len(t1_norm), len(t2_norm))
    if max_len == 0:
        return 100.0

    dist = levenshtein_distance(list(t1_norm), list(t2_norm))
    sim = max(0.0, 1.0 - (dist / max_len))
    return round(sim * 100.0, 2)


def compute_pattern_accuracy(reference: str, hypothesis: str, pattern: re.Pattern) -> Optional[float]:
    """Đo độ chính xác trích xuất thực thể theo Regex (số, ngày, tiền, mã)."""
    ref_matches = Counter(pattern.findall(reference))
    hyp_matches = Counter(pattern.findall(hypothesis))

    if not ref_matches:
        return None  # Không có thực thể này trong tài liệu chuẩn

    matched = 0
    total = sum(ref_matches.values())
    for item, count in hyp_matches.items():
        if item in ref_matches:
            matched += min(count, ref_matches[item])

    return matched / total if total > 0 else 1.0


def compute_diacritic_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    """
    Diacritic Error Rate cho dấu tiếng Việt.
    Đo tỷ lệ các nguyên âm tiếng Việt có dấu trong reference bị mất dấu hoặc sai dấu trong hypothesis.
    """
    ref_vowels = [c for c in reference if c in VIETNAMESE_VOWELS_ACCENTED]
    if not ref_vowels:
        return None

    # Tách dấu ra để so sánh
    hyp_vowels = [c for c in hypothesis if c in VIETNAMESE_VOWELS_ACCENTED]
    ref_counts = Counter(ref_vowels)
    hyp_counts = Counter(hyp_vowels)

    matched = sum(min(count, hyp_counts.get(c, 0)) for c, count in ref_counts.items())
    return (len(ref_vowels) - matched) / len(ref_vowels)


@dataclass
class PageComparisonMetric:
    """Kết quả tính toán so sánh cho một trang."""
    document_id: str
    relative_path: str
    category: str
    page_number: int
    apple_status: str
    tesseract_status: str
    apple_chars: int
    tesseract_chars: int
    apple_words: int
    tesseract_words: int
    pairwise_char_distance: int
    pairwise_word_distance: int
    pairwise_similarity_pct: float
    char_count_diff: int
    word_count_diff: int

    # Ground truth metrics (None nếu không có ground truth)
    ground_truth_available: bool = False
    ground_truth_chars: Optional[int] = None
    apple_cer_pct: Optional[float] = None
    tesseract_cer_pct: Optional[float] = None
    apple_wer_pct: Optional[float] = None
    tesseract_wer_pct: Optional[float] = None
    apple_f1_pct: Optional[float] = None
    tesseract_f1_pct: Optional[float] = None
    winner: Optional[str] = None  # 'apple' | 'tesseract' | 'tie' | 'both_poor' | None

    # Business metrics
    digit_accuracy_apple: Optional[float] = None
    digit_accuracy_tesseract: Optional[float] = None
    date_accuracy_apple: Optional[float] = None
    date_accuracy_tesseract: Optional[float] = None
    diacritic_error_apple: Optional[float] = None
    diacritic_error_tesseract: Optional[float] = None

    quality_tags: List[str] = field(default_factory=list)


def evaluate_page_pair(
    doc_id: str,
    relative_path: str,
    category: str,
    page_number: int,
    apple_text: str,
    tesseract_text: str,
    apple_status: str,
    tesseract_status: str,
    ground_truth_text: Optional[str] = None,
    quality_tags: Optional[List[str]] = None,
    cer_threshold: float = 0.05,
) -> PageComparisonMetric:
    """
    Tính toán toàn diện các metrics cho một cặp trang:
    - Nếu KHÔNG có ground_truth_text: tính pairwise difference, KHÔNG gán winner và CER.
    - Nếu CÓ ground_truth_text: tính CER, WER, F1 và xác định winner.
    """
    a_norm = normalize_text(apple_text, "normalized")
    t_norm = normalize_text(tesseract_text, "normalized")

    a_chars = len(a_norm)
    t_chars = len(t_norm)
    a_words = len(a_norm.split())
    t_words = len(t_norm.split())

    # 1. Pairwise metrics
    p_char_dist = levenshtein_distance(list(a_norm), list(t_norm))
    p_word_dist = levenshtein_distance(a_norm.split(), t_norm.split())
    p_sim = compute_similarity(a_norm, t_norm)

    metric = PageComparisonMetric(
        document_id=doc_id,
        relative_path=relative_path,
        category=category,
        page_number=page_number,
        apple_status=apple_status,
        tesseract_status=tesseract_status,
        apple_chars=a_chars,
        tesseract_chars=t_chars,
        apple_words=a_words,
        tesseract_words=t_words,
        pairwise_char_distance=p_char_dist,
        pairwise_word_distance=p_word_dist,
        pairwise_similarity_pct=p_sim,
        char_count_diff=abs(a_chars - t_chars),
        word_count_diff=abs(a_words - t_words),
        ground_truth_available=False,
        quality_tags=quality_tags or [],
    )

    # 2. Ground Truth metrics (nếu có)
    if ground_truth_text is not None and ground_truth_text.strip():
        metric.ground_truth_available = True
        gt_norm = normalize_text(ground_truth_text, "normalized")
        metric.ground_truth_chars = len(gt_norm)

        # CER & WER
        a_cer = compute_cer(gt_norm, a_norm)
        t_cer = compute_cer(gt_norm, t_norm)
        a_wer = compute_wer(gt_norm, a_norm)
        t_wer = compute_wer(gt_norm, t_norm)

        # Multiset F1
        _, _, a_f1 = compute_multiset_word_f1(gt_norm, a_norm)
        _, _, t_f1 = compute_multiset_word_f1(gt_norm, t_norm)

        metric.apple_cer_pct = round(a_cer * 100.0, 2)
        metric.tesseract_cer_pct = round(t_cer * 100.0, 2)
        metric.apple_wer_pct = round(a_wer * 100.0, 2)
        metric.tesseract_wer_pct = round(t_wer * 100.0, 2)
        metric.apple_f1_pct = round(a_f1 * 100.0, 2)
        metric.tesseract_f1_pct = round(t_f1 * 100.0, 2)

        # Business metrics
        metric.digit_accuracy_apple = compute_pattern_accuracy(gt_norm, a_norm, REGEX_DIGITS)
        metric.digit_accuracy_tesseract = compute_pattern_accuracy(gt_norm, t_norm, REGEX_DIGITS)
        metric.date_accuracy_apple = compute_pattern_accuracy(gt_norm, a_norm, REGEX_DATE)
        metric.date_accuracy_tesseract = compute_pattern_accuracy(gt_norm, t_norm, REGEX_DATE)
        metric.diacritic_error_apple = compute_diacritic_error_rate(gt_norm, a_norm)
        metric.diacritic_error_tesseract = compute_diacritic_error_rate(gt_norm, t_norm)

        # Xác định Winner theo quy định mục 10.2
        # Cả hai cùng kém nếu CER > 40% (0.40)
        if a_cer > 0.40 and t_cer > 0.40:
            metric.winner = "both_poor"
        elif abs(a_cer - t_cer) <= cer_threshold:
            metric.winner = "tie"
        elif a_cer < t_cer:
            metric.winner = "apple"
        else:
            metric.winner = "tesseract"

    return metric


def calculate_distribution_stats(values: List[float]) -> Dict[str, float]:
    """Tính các giá trị thống kê phân phối: Mean, Median, P90, P95."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}

    s_vals = sorted(values)
    n = len(s_vals)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s_vals[int(k)]
        return s_vals[int(f)] * (c - k) + s_vals[int(c)] * (k - f)

    return {
        "mean": round(sum(s_vals) / n, 2),
        "median": round(percentile(0.50), 2),
        "p90": round(percentile(0.90), 2),
        "p95": round(percentile(0.95), 2),
    }
