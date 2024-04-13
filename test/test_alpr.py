"""
Test ALPR end-to-end.
"""

from pathlib import Path

import cv2
import pytest

from alpr.alpr import ALPR

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


@pytest.mark.parametrize("img_path, expected_plates", [(ASSETS_DIR / "prueba.jpg", {"AB123CD"})])
def test_end_to_end(img_path: Path, expected_plates: set[str]) -> None:
    im = cv2.imread(str(img_path))
    alpr = ALPR(
        {
            "resolucion_detector": 512,
            "confianza_detector": 0.25,
            "confianza_avg_ocr": 0.4,
            "confianza_low_ocr": 0.35,
        }
    )
    actual_plates = set(alpr.predict(im))
    assert actual_plates == expected_plates
