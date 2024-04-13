"""
ALPR module.
"""

from pathlib import Path
from timeit import default_timer as timer

import cv2
import numpy as np

from alpr.detector import PlateDetector
from alpr.ocr import PlateOCR


class ALPR:
    def __init__(self, cfg: dict):
        input_size = cfg["resolucion_detector"]
        if input_size not in (384, 512, 608):
            raise ValueError("Modelo detector no existe! Opciones { 384, 512, 608 }")
        detector_path = (
            Path(__file__).resolve().parent
            / "models"
            / "detection"
            / f"tf-yolo_tiny_v4-{input_size}x{input_size}-custom-anchors"
        )
        assert detector_path.is_dir()
        self.detector = PlateDetector(
            str(detector_path), input_size, score=cfg["confianza_detector"]
        )
        self.ocr = PlateOCR(cfg["confianza_avg_ocr"], cfg["confianza_low_ocr"])

    def predict(self, frame: np.ndarray) -> list:
        """
        Devuelve todas las patentes reconocidas
        a partir de un frame. Si self.guardar_bd = True
        entonces cada n patentes se guardan en la base de datos

        Parametros:
            frame: np.ndarray sin procesar (Colores en orden: RGB)
        Returns:
            Una lista con todas las patentes reconocidas
        """
        # Preprocess
        input_img = self.detector.preprocess(frame)
        # Inference
        yolo_out = self.detector.predict(input_img)
        # Bounding Boxes despues de NMS
        bboxes = self.detector.procesar_salida_yolo(yolo_out)
        # Hacer OCR a cada patente localizada
        iter_coords = self.detector.yield_coords(frame, bboxes)
        patentes = self.ocr.predict(iter_coords, frame)
        return patentes

    def mostrar_predicts(self, frame: np.ndarray):
        """
        Mostrar localizador + reconocedor

        Parametros:
            frame: np.ndarray sin procesar (Colores en orden: RGB)
        Returns:
            frame con el bounding box de la patente y
            la prediccion del texto de la patente

            total_time: tiempo de inferencia sin contar el dibujo
            de los rectangulos
        """
        # pylint: disable-msg=too-many-locals
        total_time = 0.0
        start = timer()
        # Preprocess
        input_img = self.detector.preprocess(frame)
        # Inference
        yolo_out = self.detector.predict(input_img)
        # Bounding Boxes despues de NMS
        bboxes = self.detector.procesar_salida_yolo(yolo_out)
        # Hacer y mostrar OCR
        iter_coords = self.detector.yield_coords(frame, bboxes)
        end = timer()
        total_time += end - start
        font_scale = 1.25
        for yolo_prediction in iter_coords:
            x1, y1, x2, y2, _ = yolo_prediction
            #
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            #
            start = timer()
            plate, probs = self.ocr.predict_ocr(x1, y1, x2, y2, frame)
            total_time += timer() - start
            avg = np.mean(probs)
            if avg > self.ocr.confianza_avg and self.ocr.none_low(
                probs, thresh=self.ocr.none_low_thresh
            ):
                plate = ("".join(plate)).replace("_", "")
                mostrar_txt = f"{plate} {avg * 100:.2f}%"
                cv2.putText(
                    img=frame,
                    text=mostrar_txt,
                    org=(x1 - 20, y1 - 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=[0, 0, 0],
                    lineType=cv2.LINE_AA,
                    thickness=6,
                )
                cv2.putText(
                    img=frame,
                    text=mostrar_txt,
                    org=(x1 - 20, y1 - 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale,
                    color=[255, 255, 255],
                    lineType=cv2.LINE_AA,
                    thickness=2,
                )
        return frame, total_time
