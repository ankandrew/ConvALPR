import string

import cv2
import numpy as np
import tensorflow as tf

MODELOS = {
    1: 'alpr/models/ocr/m1_2.0M_GPU',
    2: 'alpr/models/ocr/m2_1.5M_GPU',
    3: 'alpr/models/ocr/m3_1.3M_CPU',
    4: 'alpr/models/ocr/m4_1.1M_CPU',
}


class PlateOCR:
    """
    Modulo encargado del reconocimiento
    de caracteres de las patentes (ya recortadas)
    """

    def __init__(self,
                 ocr_model_num: int = 4,
                 confianza_avg: float = 0.5,
                 none_low_thresh: float = 0.35):
        """
        Parametros:
            ocr_model_num   Numero del modelo a usar (1-4)
        """
        if ocr_model_num not in MODELOS:
            raise KeyError('Modelo inexistente, valores posibles: (1-4)')

        ocr_model_path = MODELOS[ocr_model_num]
        self.imported = tf.saved_model.load(ocr_model_path)
        self.cnn_ocr_model = self.imported.signatures["serving_default"]
        self.alphabet = string.digits + string.ascii_uppercase + '_'
        self.confianza_avg = confianza_avg
        self.none_low_thresh = none_low_thresh

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
            if avg > self.confianza_avg and self.none_low(probs, thresh=self.none_low_thresh):
                plate = (''.join(plate)).replace('_', '')
                patentes.append(plate)
        return patentes

    def none_low(self, probs, thresh=.5):
        """
        Devuelve False si hay algun caracter
        con probabilidad por debajo de thresh
        """
        for prob in probs:
            if prob < thresh:
                return False
        return True

    def print_plates(self):
        print(', '.join(self.unique_plates))

    def predict_ocr(self,
                    x1: int,
                    y1: int,
                    x2: int,
                    y2: int,
                    frame: np.ndarray):
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
        prediction_ocr = self.__predict_from_array(cropped_plate)
        plate, probs = self.__probs_to_plate(prediction_ocr)
        return plate, probs

    def __probs_to_plate(self, prediction):
        prediction = prediction.reshape((7, 37))
        probs = np.max(prediction, axis=-1)
        prediction = np.argmax(prediction, axis=-1)
        plate = list(map(lambda x: self.alphabet[x], prediction))
        return plate, probs

    def __predict_from_array(self, patente_recortada: np.ndarray):
        """
        Hace el preprocessing (normaliza, agrega batch_dimension)
        y hace la inferencia

        Parametros:
            patente_recortada: array conteniendo la imagen recortada (solo la patente)
        Returns:
            np.array de (1,259) que contiene las predicciones para cada
            caracter de la patente (37 posibles caracteres * 7 lugares)
        """
        patente_recortada = cv2.cvtColor(patente_recortada, cv2.COLOR_RGB2GRAY)
        patente_recortada = cv2.resize(patente_recortada, (140, 70))
        patente_recortada = patente_recortada[np.newaxis, ..., np.newaxis]
        patente_recortada = tf.constant(
            patente_recortada, dtype=tf.float32) / 255.
        # Hacer prediction
        pred = self.cnn_ocr_model(patente_recortada)
        return pred[next(iter(pred))].numpy()
