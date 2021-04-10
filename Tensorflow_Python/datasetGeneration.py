import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


def test_data_generator(
    data_dir, mode, target_size=(256, 256), batch_size=32, shuffle=True
):
    for imgs in ImageDataGenerator().flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle,
    ):
        yield imgs / 255.0

def createDataset(imageSet_a, imageSet_b):
        yield imageSet_a, imageSet_b

loadedImages = next(
    test_data_generator(
        "./", "Images", batch_size=96, shuffle=False
    )
)
loadedPixelart = next(
    test_data_generator(
        "./", "Images", batch_size=96, shuffle=False
    )
)

train_x, train_y = next(
    createDataset(
        loadedImages, loadedPixelart
    )
)

print(str(len(loadedImages)))
print(str(len(loadedPixelart)))
print(str(len(train_x)))
print(str(len(train_y)))