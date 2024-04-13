"""
Plate Detector.
"""

import cv2
import numpy as np
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class PlateDetector:
    """
    Modulo encargado del detector de patentes
    """

    def __init__(
        self, weights_path: str, input_size: int = 608, iou: float = 0.45, score: float = 0.25
    ):
        self.input_size = input_size
        self.iou = iou
        self.score = score
        self.saved_model_loaded = tf.saved_model.load(weights_path)
        self.yolo_infer = self.saved_model_loaded.signatures["serving_default"]

    def procesar_salida_yolo(self, output):
        """
        Modificado de https://github.com/hunglc007/tensorflow-yolov4-tflite -
                      /blob/9f16748aa3f45ff240608da4bd9b1216a29127f5/detectvideo.py#L91
        Aplica a la salida de yolo Non Max Suppression (NMS) eliminando detecciones
        duplicadas del mismo objeto

        Parametros:
            output: tensor con la salida de yolo
        Returns:
            Lista con losd Bounding Boxes de todas
            las patentes detectadas despues de NMS
        """
        for value in output.values():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score,
        )
        return [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

    def preprocess(self, frame):
        """
        Normalizar pixeles entre [0; 1], agregar
        batch dimension y resizear la imagen para el
        el modelo correspondiente de yolo
        ej de las dimensiones : (1920,1080,3) -> (1,608,608,3)
        Parametros
            frame: numpy array
        Returns: tf Tensor preprocesado
        """
        image_data = cv2.resize(frame, (self.input_size, self.input_size))
        image_data = image_data / 255.0
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return tf.constant(image_data)

    def predict(self, input_img: tf.Tensor):
        """
        Hace la inferencia a partir del tensor
        que contiene la img de entrada

        Parametros:
            input_img: tf Tensor con dimensiones
            (1, self.input_size, self.input_size, 3)
        Returns:
            Output de la salida de YOLO
        """
        return self.yolo_infer(input_img)

    def draw_bboxes(self, frame: np.ndarray, bboxes: list):
        """
        Para visualizar la salida del detector, se dibujan
        todos los rectangulos correspondiente a las patentes
        en el frame de entrada

        Parametros:
            frame: numpy array conteniendo el frame original
            bboxes: predicciones/output de despues del NMS
        Returns:
            Numpy array conteniendo los rectangulos dibujados
        """
        for x1, y1, x2, y2, score in self.yield_coords(frame, bboxes):
            font_scale = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(
                frame,
                f"{score:.2f}%",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (20, 10, 220),
                5,
            )
        return frame

    def yield_coords(self, frame: np.ndarray, bboxes: list):
        """
        Devuelve cada coordenada de los
        rectangulo localizadas

        Parametros:
            frame: numpy array conteniendo el frame original
            bboxes: predicciones/output de despues del NMS

        Returns:
            (x1, y1, x2, y2):   Coordenas relativas al frame de la determinada
                                patente
            score:  Probabilidad de objectness de yolo
                    (que tan seguro piensa que es una patente)
        """
        out_boxes, out_scores, _, num_boxes = bboxes
        image_h, image_w, _ = frame.shape
        for i in range(num_boxes[0]):
            coor = out_boxes[0][i]
            x1 = int(coor[1] * image_w)
            y1 = int(coor[0] * image_h)
            x2 = int(coor[3] * image_w)
            y2 = int(coor[2] * image_h)
            yield x1, y1, x2, y2, out_scores[0][i]
