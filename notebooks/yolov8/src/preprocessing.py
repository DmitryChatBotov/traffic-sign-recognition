import cv2
import numpy as np


def _normalize(image: np.ndarray) -> np.ndarray:
    return image / 255.0


def _resize(
    image: np.ndarray, new_image_size: tuple[int, int], fill_color=(0, 0, 0)
) -> np.ndarray:
    image_height, image_width, *_ = image.shape
    new_image_width, new_image_height = new_image_size
    ratio = min(new_image_width / image_width, new_image_height / image_height)

    scaled_image_width = int(round(image_width * ratio))
    scaled_image_height = int(round(image_height * ratio))
    if (image_width, image_height) == (scaled_image_width, scaled_image_height):
        return image

    image = cv2.resize(
        image,
        (scaled_image_width, scaled_image_height),
        interpolation=cv2.INTER_LINEAR,
    )

    width_padding = (new_image_width - scaled_image_width) / 2
    height_padding = (new_image_height - scaled_image_height) / 2

    top, bottom = int(round(height_padding - 0.1)), int(round(height_padding + 0.1))
    left, right = int(round(width_padding - 0.1)), int(round(width_padding + 0.1))

    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color
    )


def _transpose(image: np.ndarray) -> np.ndarray:
    return image.transpose(2, 0, 1)


def _add_batch(image: np.ndarray) -> np.ndarray:
    return image[np.newaxis, :, :, :].astype(np.float32)


def preprocess_image(image: np.ndarray, image_size: tuple[int, int]):
    img_height, img_width = image.shape[:2]

    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_img = _normalize(input_img)
    input_img = _resize(image=input_img, new_image_size=image_size)
    input_img = _transpose(input_img)
    input_img = _add_batch(input_img)

    return input_img, img_width, img_height
