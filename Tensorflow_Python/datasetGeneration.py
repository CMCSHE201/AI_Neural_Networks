import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


def loadImageSet(
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
    loadImageSet(
        "./", "images_normal", target_size=(1024,1024), batch_size=95, shuffle=False
    )
)
loadedPixelart = next(
    loadImageSet(
        "./", "images_pixel", target_size=(128,128), batch_size=95, shuffle=False
    )
)

train_x, test_x, train_y, test_y = train_test_split( np.array(loadedImages) , np.array(loadedPixelart) , test_size=0.1 )

print(str(len(loadedImages)))
print(str(len(loadedPixelart)))

print(str(len(train_x)))
print(str(len(train_y)))

print(str(len(test_x)))
print(str(len(test_y)))