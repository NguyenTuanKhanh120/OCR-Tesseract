import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tesseract_ocr.document_renderer import sha256_file
from tesseract_ocr.insurance_pipeline import (
    InsurancePipelineConfig,
    run_insurance_region_pipeline,
)


class InsurancePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pdf = self.root / "input" / "sample.pdf"
        self.pdf.parent.mkdir(parents=True)
        self.pdf.write_bytes(b"%PDF-1.4\nmock\n")
        self.run_dir = self.root / "run"
        self.tessdata = self.root / "tessdata"
        self.tessdata.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_runs_all_seven_steps_in_order(self):
        calls = []
        progress = []

        def stage(name, manifest_name, result):
            def implementation(input_path, output_dir, *args, **kwargs):
                calls.append(name)
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / manifest_name).write_text("{}", encoding="utf-8")
                return dict(result)

            return implementation

        def merge(region_manifest, output_dir, source_pdf=None, **kwargs):
            calls.append("merge")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / "sample_ocr.txt"
            output.write_text("--- [TRANG 1/1] ---\nNội dung\n", encoding="utf-8")
            (output_dir / "merge_manifest.json").write_text("{}", encoding="utf-8")
            return {
                "status": "success",
                "page_count": 1,
                "panel_count": 2,
                "needs_review_panels": 0,
                "output_text_file": str(output),
            }

        with (
            patch(
                "tesseract_ocr.insurance_pipeline.render_pdf_to_pngs",
                side_effect=stage(
                    "render", "render_manifest.json", {"page_count": 1}
                ),
            ),
            patch(
                "tesseract_ocr.insurance_pipeline.split_rendered_pages",
                side_effect=stage(
                    "split",
                    "panels_manifest.json",
                    {"page_count": 1, "panel_count": 2},
                ),
            ),
            patch(
                "tesseract_ocr.insurance_pipeline.rectify_panels",
                side_effect=stage(
                    "rectify",
                    "perspective_manifest.json",
                    {"page_count": 1, "panel_count": 2},
                ),
            ),
            patch(
                "tesseract_ocr.insurance_pipeline.mask_panel_artifacts",
                side_effect=stage(
                    "mask",
                    "artifact_mask_manifest.json",
                    {"page_count": 1, "panel_count": 2},
                ),
            ),
            patch(
                "tesseract_ocr.insurance_pipeline.process_masked_panels",
                side_effect=stage(
                    "background",
                    "background_manifest.json",
                    {"page_count": 1, "panel_count": 2},
                ),
            ),
            patch(
                "tesseract_ocr.insurance_pipeline.ocr_preprocessed_panels",
                side_effect=stage(
                    "ocr",
                    "region_ocr_manifest.json",
                    {
                        "page_count": 1,
                        "panel_count": 2,
                        "candidate_run_count": 12,
                    },
                ),
            ),
            patch(
                "tesseract_ocr.insurance_pipeline.merge_region_ocr_results",
                side_effect=merge,
            ),
        ):
            result = run_insurance_region_pipeline(
                self.pdf,
                self.run_dir,
                self.tessdata,
                progress_callback=lambda step, name: progress.append((step, name)),
            )

        self.assertEqual(
            calls,
            ["render", "split", "rectify", "mask", "background", "ocr", "merge"],
        )
        self.assertEqual([item[0] for item in progress], list(range(1, 8)))
        self.assertEqual(result["page_count"], 1)
        self.assertEqual(result["panel_count"], 2)
        self.assertEqual(result["candidate_run_count"], 12)
        self.assertEqual(len(result["stages"]), 7)
        self.assertTrue((self.run_dir / "pipeline_manifest.json").is_file())

    def test_reuses_verified_completed_run(self):
        output = self.run_dir / "step7_merged" / "sample_ocr.txt"
        output.parent.mkdir(parents=True)
        output.write_text("Kết quả", encoding="utf-8")
        config = InsurancePipelineConfig()
        cached = {
            "source_pdf": str(self.pdf.resolve()),
            "source_pdf_sha256": sha256_file(self.pdf),
            "config": config.to_dict(),
            "output_text_file": str(output),
            "output_text_sha256": sha256_file(output),
        }
        (self.run_dir / "pipeline_manifest.json").write_text(
            json.dumps(cached, ensure_ascii=False), encoding="utf-8"
        )

        with patch(
            "tesseract_ocr.insurance_pipeline.render_pdf_to_pngs"
        ) as render_mock:
            result = run_insurance_region_pipeline(
                self.pdf, self.run_dir, self.tessdata, config=config
            )

        render_mock.assert_not_called()
        self.assertTrue(result["cache_hit"])

    def test_rejects_invalid_expected_panel_count(self):
        with self.assertRaisesRegex(ValueError, "expected_panels"):
            run_insurance_region_pipeline(
                self.pdf,
                self.run_dir,
                self.tessdata,
                config=InsurancePipelineConfig(expected_panels=4),
            )


if __name__ == "__main__":
    unittest.main()
