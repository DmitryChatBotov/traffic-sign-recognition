from pathlib import Path

import numpy as np
import cv2
import onnxruntime
import onnxruntime as nxrun
from preprocessing import preprocess_image
from postprocessing import process_output


class TrafficSignDetector:
    def __init__(self, path: Path | str, conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._model = self._load_model(path=path)
        self._model_input_names, self._input_width, self._input_height = self._get_model_input()
        self._model_output_names = self._get_model_output()

    def __call__(self, image: np.ndarray):
        return self._detect(image)

    @staticmethod
    def _load_model(path: Path | str) -> nxrun.InferenceSession:
        return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider", "CUDAExecutionProvider"])

    def _get_model_input(self):
        model_inputs = self._model.get_inputs()
        model_input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        input_width = 640
        input_height = 480

        return model_input_names, input_width, input_height

    def _get_model_output(self):
        return [model_output.name for model_output in self._model.get_outputs()]

    def _infer(self, input: np.ndarray):
        return self._model.run(self._model_output_names, {self._model_input_names[0]: input})

    def _detect(self, image: np.ndarray):
        print(self._input_width)
        model_input, img_width, img_height = preprocess_image(image, (self._input_width, self._input_height))
        outputs = self._infer(model_input)
        bboxes, scores, class_names = process_output(outputs, self._conf_threshold, self._iou_threshold,
                                                     self._input_width, self._input_height, img_width, img_height)
        return bboxes, scores, class_names


if __name__ == "__main__":
    detector = TrafficSignDetector('../best.onnx')
    for filename in Path('../../data').iterdir():
        image = cv2.imread(str(filename))
        print(filename)
        print(detector(image))