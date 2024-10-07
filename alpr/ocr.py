"""
OCR module.
"""

import cv2
import numpy as np
from fast_plate_ocr import ONNXPlateRecognizer
from open_image_models.detection.core.base import DetectionResult


class PlateOCR:
    """
    Modulo encargado del reconocimiento
    de caracteres de las patentes (ya recortadas)
    """

    def __init__(self, confianza_avg: float = 0.5, none_low_thresh: float = 0.35) -> None:
        self.confianza_avg = confianza_avg
        self.none_low_thresh = none_low_thresh
        self.ocr_model = ONNXPlateRecognizer("argentinian-plates-cnn-model")

    def predict(self, plate_detections: list[DetectionResult], frame: np.ndarray) -> list:
        """
        Reconoce a partir de un frame todas
        las patentes en formato de texto

        Parametros:
            plate_detections: Predicciones del detector de License plates.
            frame:  sub-frame conteniendo la patente candidato
        Returns:
            Lista de patentes (en formato de texto)
        """
        patentes = []
        for yolo_prediction in plate_detections:
            # pylint: disable=duplicate-code
            x1, y1, x2, y2 = (
                yolo_prediction.bounding_box.x1,
                yolo_prediction.bounding_box.y1,
                yolo_prediction.bounding_box.x2,
                yolo_prediction.bounding_box.y2,
            )
            plate, probs = self.predict_ocr(x1, y1, x2, y2, frame)
            if plate is None or probs is None:
                continue
            # Ignorar si tiene baja confianza el OCR
            avg = np.mean(probs)
            if avg > self.confianza_avg and self.none_low(probs[0], thresh=self.none_low_thresh):
                plate = ("".join(plate)).replace("_", "")
                patentes.append(plate)
        return patentes

    def none_low(self, probs, thresh=0.5):
        """
        Devuelve False si hay algun caracter
        con probabilidad por debajo de thresh
        """
        return all(prob >= thresh for prob in probs.tolist())

    def predict_ocr(self, x1: int, y1: int, x2: int, y2: int, frame: np.ndarray):
        """
        Hace OCR en un sub-frame del frame

        Parametros:
            x1: Valor de x de la esquina superior izquierda del rectangulo
            y1:    "     y           "             "                  "
            x2: Valor de x de la esquina inferior derecha del rectangulo
            y2:    "     y           "             "                  "
            frame: array conteniendo la imagen original
        """
        cropped_plate = frame[y1:y2, x1:x2]
        if cropped_plate is None:
            return None, None
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        plate, probs = self.ocr_model.run(cropped_plate, return_confidence=True)
        return plate, probs
