"""
ALPR module.
"""

from timeit import default_timer as timer

import cv2
import numpy as np
from open_image_models import LicensePlateDetector
from open_image_models.detection.core.hub import PlateDetectorModel

from alpr.ocr import PlateOCR


class ALPR:
    def __init__(self, cfg: dict):
        input_size = cfg["resolucion_detector"]
        detection_model: PlateDetectorModel
        if input_size == 256:
            detection_model = "yolo-v9-t-256-license-plate-end2end"
        elif input_size == 384:
            detection_model = "yolo-v9-t-384-license-plate-end2end"
        elif input_size == 512:
            detection_model = "yolo-v9-t-512-license-plate-end2end"
        elif input_size == 640:
            detection_model = "yolo-v9-t-640-license-plate-end2end"
        else:
            raise ValueError("Modelo detector no existe! Opciones { 256, 384, 512, 640 }")
        self.detector = LicensePlateDetector(
            detection_model=detection_model, conf_thresh=cfg["confianza_detector"]
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
        # Run detection
        plate_detections = self.detector.predict(frame)
        # Run OCR
        patentes = self.ocr.predict(plate_detections, frame)
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
        # Run detection
        plate_detections = self.detector.predict(frame)
        end = timer()
        total_time += end - start
        font_scale = 1.25
        for yolo_prediction in plate_detections:  # pylint: disable=not-an-iterable
            x1, y1, x2, y2 = (
                yolo_prediction.bounding_box.x1,
                yolo_prediction.bounding_box.y1,
                yolo_prediction.bounding_box.x2,
                yolo_prediction.bounding_box.y2,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            start = timer()
            plate, probs = self.ocr.predict_ocr(x1, y1, x2, y2, frame)
            if plate is None or probs is None:
                continue
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
