from copy import deepcopy

import cv2
import numpy as np

CLASS_NAMES = [
    "2_1",
    "1_23",
    "1_17",
    "3_24",
    "8_2_1",
    "5_20",
    "5_19_1",
    "5_16",
    "3_25",
    "6_16",
    "2_2",
    "2_4",
    "8_13_1",
    "4_2_1",
    "1_20_3",
    "1_25",
    "3_4",
    "8_3_2",
    "3_4_1",
    "4_1_6",
    "4_2_3",
    "4_1_1",
    "1_33",
    "5_15_5",
    "3_27",
    "1_15",
    "4_1_2_1",
    "6_3_1",
    "8_1_1",
    "6_7",
    "5_15_3",
    "7_3",
    "1_19",
    "6_4",
    "8_1_4",
    "1_16",
    "1_11_1",
    "6_6",
    "5_15_1",
    "7_2",
    "5_15_2",
    "7_12",
    "3_18",
    "5_6",
    "5_5",
    "7_4",
    "4_1_2",
    "8_2_2",
    "7_11",
    "1_22",
    "1_27",
    "2_3_2",
    "5_15_2_2",
    "1_8",
    "3_13",
    "2_3",
    "2_3_3",
    "7_7",
    "1_11",
    "8_13",
    "1_12_2",
    "1_20",
    "1_12",
    "3_32",
    "2_5",
    "3_1",
    "4_8_2",
    "3_20",
    "3_2",
    "5_22",
    "7_5",
    "8_4_1",
    "3_14",
    "1_2",
    "1_20_2",
    "4_1_4",
    "7_6",
    "8_3_1",
    "4_3",
    "4_1_5",
    "8_2_3",
    "8_2_4",
    "3_10",
    "4_2_2",
    "7_1",
    "3_28",
    "4_1_3",
    "5_3",
    "3_31",
    "6_2",
    "1_21",
    "3_21",
    "1_13",
    "1_14",
    "6_15_2",
    "2_6",
    "3_18_2",
    "4_1_2_2",
    "3_19",
    "8_5_4",
    "5_15_7",
    "5_14",
    "5_21",
    "1_1",
    "6_15_1",
    "8_6_4",
    "8_15",
    "3_11",
    "3_30",
    "5_7_1",
    "5_7_2",
    "1_5",
    "3_29",
    "5_11",
    "3_12",
    "5_8",
    "8_5_2",
]


def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def plot_detection_result(image: np.ndarray, bboxes) -> np.ndarray:
    for bbox in bboxes:
        cv2.rectangle(image, bbox[:2], bbox[2:4], (255, 255, 0), 2)
        cv2.putText(
            image,
            f"{bbox[4]}:{bbox[5]}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2,
        )

    return image


def label_video(input_video_path, output_video_path, detector):
    """
    Processes a video using the given processing_function and saves the result.

    Args:
    input_video_path (str): Path to the input video.
    output_video_path (str): Path where the processed video will be saved.
    """

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get properties from the input video for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' for .mp4 format

    # Create a VideoWriter object to write the video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Read and process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions = detector(frame)
        outputs = []
        for prediction in predictions:
            box = prediction["bbox"]
            x1, y1, x2, y2 = [round(x) for x in box]
            class_name = prediction["class"]
            prob = round(prediction["score"], 2)
            outputs.append([x1, y1, x2, y2, class_name, prob])

        processed_frame = plot_detection_result(deepcopy(frame), outputs)

        out.write(processed_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
