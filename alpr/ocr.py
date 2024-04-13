"""
OCR module.
"""

import cv2
import numpy as np
from fast_plate_ocr import ONNXPlateRecognizer


class PlateOCR:
    """
    Modulo encargado del reconocimiento
    de caracteres de las patentes (ya recortadas)
    """

    def __init__(self, confianza_avg: float = 0.5, none_low_thresh: float = 0.35) -> None:
        self.confianza_avg = confianza_avg
        self.none_low_thresh = none_low_thresh
        self.ocr_model = ONNXPlateRecognizer("argentinian-plates-cnn-model")

    def predict(self, iter_coords, frame: np.ndarray) -> list:
        """
        Reconoce a partir de un frame todas
        las patentes en formato de texto

        Parametros:
            iter_coords:    generator object que yieldea las patentes
            frame:  sub-frame conteniendo la patente candidato
        Returns:
            Lista de patentes (en formato de texto)
        """
        patentes = []
        for yolo_prediction in iter_coords:
            # x1, y1, x2, y2, score = yolo_prediction
            x1, y1, x2, y2, _ = yolo_prediction
            plate, probs = self.predict_ocr(x1, y1, x2, y2, frame)
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
        return all(prob >= thresh for prob in probs)

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
        cropped_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        plate, probs = self.ocr_model.run(cropped_plate, return_confidence=True)
        return plate, probs
