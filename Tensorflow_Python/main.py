import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import numpy as np
from utils import psnr

import os

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter


# ========================================================================================================================================================================================================
# Defining a specific layer for handling data augmentation
# ========================================================================================================================================================================================================
def loadImageSet(
    data_dir, mode, target_size=(256, 256), batch_size=5, shuffle=True
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

loadedPixelart = next(
    loadImageSet(
        "./", "images_pixel", target_size=(128,128), batch_size=233, shuffle=False)
    )
loadedImages = next(
    loadImageSet(
        "./", "images_normal", target_size=(1024,1024), batch_size=233, shuffle=False)
    )

train_x, test_x, train_y, test_y = train_test_split( np.array(loadedPixelart) , np.array(loadedImages) , test_size=0.1 )


# ========================================================================================================================================================================================================
# Defining a specific layer for handling data augmentation
# ========================================================================================================================================================================================================
# Allows the network to creates 4 variations on the data by either flipping the image on either the x or y axis or both or none
data_augmentation = tf.keras.Sequential([
layers.experimental.preprocessing.RandomFlip("horizontal"),
layers.experimental.preprocessing.RandomFlip("vertical")
])


# ========================================================================================================================================================================================================
# Defining the Neural Network
# ========================================================================================================================================================================================================
inputs = tf.keras.Input(shape=(None, None, 3))
# Data Augmentation
x = data_augmentation(inputs)
# Convolutional Base
x = layers.Conv2D(filters=64, kernel_size=9, activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(filters=32, kernel_size=1, activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Dropout(0.1)(x)
x = layers.Conv2D(filters=3, kernel_size=5, padding="same")(x)
outputs = layers.UpSampling2D((2, 2))(x)
model = tf.keras.Model(inputs, outputs)
# Displaying the model architecture
model.summary()
    

# ========================================================================================================================================================================================================
# Training the Model
# ========================================================================================================================================================================================================
# Compile and train the model
model.compile(loss="mean_squared_error", optimizer="adam", metrics=[psnr, 'accuracy'])
history = model.fit(
    train_x, 
    train_y,
    validation_data=(test_x, test_y),
    batch_size=5,
    epochs=100
)
# Saving the trained model (needs to be converted to an onnx format to be compatable with Unity/Barracuda
model.save('./SRCNNModel')
print("Model Saved to: " + os.path.dirname(os.path.realpath('./SRCNNModel')) + "\SRCNNModel")


# ========================================================================================================================================================================================================
# Evaluate the model
# ========================================================================================================================================================================================================
# Creating graph to show model's accuracy & valuation accuracy per epoch    
plot1 = plt.figure(1)
plt.plot(history.history['psnr'], label='psnr')
plt.plot(history.history['val_psnr'], label = 'val_psnr')
plt.xlabel('Epoch')
plt.ylabel('psnr')
plt.legend(loc='lower right')

plot2 = plt.figure(2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
# Formatting Graph Y Axis to show units as a percentage
yTicks = mtick.PercentFormatter(1, None, '%', False)
axes = plt.gca()
axes.yaxis.set_major_formatter(yTicks)
plt.legend(loc='lower right')

# Printing out final model accuracy
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print(test_acc)
# Showing graph in seperate window
plt.show()