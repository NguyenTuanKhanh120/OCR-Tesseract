import tempfile
import unittest
from pathlib import Path

from tesseract_ocr.tesseract_engine import find_tessdata_directory


class TestTessdataResolution(unittest.TestCase):
    def test_requires_every_requested_language(self):
        with tempfile.TemporaryDirectory(prefix="tessdata_test_") as temp_dir:
            tessdata = Path(temp_dir)
            (tessdata / "eng.traineddata").touch()

            self.assertIsNone(
                find_tessdata_directory(
                    custom_dir=tessdata,
                    required_langs=["zzz_test", "eng"],
                )
            )

            (tessdata / "zzz_test.traineddata").touch()
            self.assertEqual(
                find_tessdata_directory(
                    custom_dir=tessdata,
                    required_langs=["zzz_test", "eng"],
                ),
                tessdata.resolve(),
            )


if __name__ == "__main__":
    unittest.main()
