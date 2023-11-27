import os
from copy import deepcopy

import cv2
import gradio as gr

from traffic_sign_detector import TrafficSignDetector
from utils import plot_detection_result


def process_video(input_video_path, output_video_path):
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
    detector = TrafficSignDetector(os.path.join(os.pardir, "models", "best.onnx"))

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


def gradio_interface(input_video_path):
    output_video_path = input_video_path.rsplit(".", 1)[0] + "_processed.mp4"
    # Обработка видео
    process_video(input_video_path, output_video_path)

    return output_video_path


iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(label="Загрузите видео с регистратора"),
    outputs=gr.Video(label="Скачать обработанное видео"),
    title="Разметка дорожных знаков на видео",
    description="Загрузите видео с регистратора, и приложение разметит все знаки дорожного движения.",
    allow_flagging=False,  # Убирает кнопку "Flag"
)

# Запуск приложения
iface.launch()
