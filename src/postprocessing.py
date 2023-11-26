from typing import Tuple

import numpy as np

from utils import CLASS_NAMES, multiclass_nms, xywh2xyxy


def postprocess_output(
    output: np.ndarray,
    input_width: int,
    input_height: int,
    img_width: int,
    img_height: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
) -> Tuple[list[int], list[float], list[str]]:
    """Возвращает результаты детекции, отсекая невалидные по трешхолдам."""
    predictions = np.squeeze(output[0]).T

    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores >= conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    class_ids = np.argmax(predictions[:, 4:], axis=1)

    boxes = _extract_boxes(
        predictions, input_width, input_height, img_width, img_height
    )

    indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)

    return boxes[indices], scores[indices], [CLASS_NAMES[i] for i in class_ids[indices]]


def _extract_boxes(
    predictions: np.ndarray,
    input_width: int,
    input_height: int,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    boxes = predictions[:, :4]
    boxes = xywh2xyxy(boxes)
    boxes = _rescale_boxes(boxes, input_width, input_height, img_width, img_height)

    return boxes


def _rescale_boxes(
    boxes: np.ndarray,
    input_width: int,
    input_height: int,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    scale = min(input_width / img_width, input_height / img_height)
    x_padding = (input_width - img_width * scale) / 2
    y_padding = (input_height - img_height * scale) / 2

    rescaled_boxes = np.empty_like(boxes)

    rescaled_boxes[:, 0] = (boxes[:, 0] - x_padding) / scale
    rescaled_boxes[:, 1] = (boxes[:, 1] - y_padding) / scale
    rescaled_boxes[:, 2] = (boxes[:, 2] - x_padding) / scale
    rescaled_boxes[:, 3] = (boxes[:, 3] - y_padding) / scale

    return rescaled_boxes
