import os

import gradio as gr

from traffic_sign_detector import TrafficSignDetector


def process_video(input_video_path):
    output_video_path = input_video_path.rsplit(".", 1)[0] + "_processed.mp4"

    detector = TrafficSignDetector(os.path.join(os.pardir, "models", "best.onnx"))
    detector.label_video(input_video_path, output_video_path)

    return output_video_path


iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Загрузите видео с регистратора"),
    outputs=gr.Video(label="Скачать обработанное видео"),
    title="Разметка дорожных знаков на видео",
    description="Загрузите видео с регистратора, и приложение разметит все знаки дорожного движения.",
    allow_flagging=False,  # Убирает кнопку "Flag"
)

# Запуск приложения
iface.launch()
