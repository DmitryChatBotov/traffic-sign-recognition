import numpy as np

from utils import class_names, multiclass_nms, xywh2xyxy


def process_output(
    output,
    conf_threshold,
    iou_threshold,
    input_width,
    input_height,
    img_width,
    img_height,
):
    predictions = np.squeeze(output[0]).T

    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores >= conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        return [], [], []

    # Get the class with the highest confidence
    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = extract_boxes(predictions, input_width, input_height, img_width, img_height)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    # indices = nms(boxes, scores, self.iou_threshold)
    indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)

    return boxes[indices], scores[indices], [class_names[i] for i in class_ids[indices]]


def extract_boxes(predictions, input_width, input_height, img_width, img_height):
    # Extract boxes from predictions
    boxes = predictions[:, :4]

    # Scale boxes to original image dimensions
    boxes = rescale_boxes(boxes, input_width, input_height, img_width, img_height)

    # Convert boxes to xyxy format
    boxes = xywh2xyxy(boxes)

    return boxes


def rescale_boxes(boxes, input_width, input_height, img_width, img_height):
    # Rescale boxes to original image dimensions
    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes
