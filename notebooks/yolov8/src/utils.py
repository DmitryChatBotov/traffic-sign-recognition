import numpy as np

class_names = [
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
    "7_15",
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
    "8_8",
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
    "8_3_3",
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
    "2_3_6",
    "5_22",
    "5_18",
    "2_3_5",
    "7_5",
    "8_4_1",
    "3_14",
    "1_2",
    "1_20_2",
    "4_1_4",
    "7_6",
    "8_1_3",
    "8_3_1",
    "4_3",
    "4_1_5",
    "8_2_3",
    "8_2_4",
    "1_31",
    "3_10",
    "4_2_2",
    "7_1",
    "3_28",
    "4_1_3",
    "5_4",
    "5_3",
    "6_8_2",
    "3_31",
    "6_2",
    "1_21",
    "3_21",
    "1_13",
    "1_14",
    "2_3_4",
    "4_8_3",
    "6_15_2",
    "2_6",
    "3_18_2",
    "4_1_2_2",
    "1_7",
    "3_19",
    "1_18",
    "2_7",
    "8_5_4",
    "5_15_7",
    "5_14",
    "5_21",
    "1_1",
    "6_15_1",
    "8_6_4",
    "8_15",
    "4_5",
    "3_11",
    "8_18",
    "8_4_4",
    "3_30",
    "5_7_1",
    "5_7_2",
    "1_5",
    "3_29",
    "6_15_3",
    "5_12",
    "3_16",
    "1_30",
    "5_11",
    "1_6",
    "8_6_2",
    "6_8_3",
    "3_12",
    "3_33",
    "8_4_3",
    "5_8",
    "8_14",
    "8_17",
    "3_6",
    "1_26",
    "8_5_2",
    "6_8_1",
    "5_17",
    "1_10",
    "8_16",
    "7_18",
    "7_14",
    "8_23",
]

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
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
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
