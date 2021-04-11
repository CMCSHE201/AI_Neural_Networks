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
    data_dir, mode, target_size=(1024, 1024), batch_size=5, shuffle=False
):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(
        directory=data_dir,
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle)
    for imgs in gen:
        yield imgs / 255.0

print(os.listdir("./pixelImages/images_pixel"))
loadedPixelart = next(
    loadImageSet(
        "./pixelImages/", "images_pixel", target_size=(1024,1024), batch_size=95, shuffle=False)
    )
print(os.listdir("./normalImages/images_normal"))
loadedImages = next(
    loadImageSet(
        "./normalImages/", "images_normal", target_size=(1024,1024), batch_size=95, shuffle=False)
    )

train_x, test_x, train_y, test_y = train_test_split( np.array(loadedPixelart) , np.array(loadedImages) , test_size=0.1 )

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(loadedPixelart[i])
  plt.axis("off")
    
plt.figure(figsize=(10, 10))
for j in range(9):
  ax = plt.subplot(3, 3, j + 1)
  plt.imshow(loadedImages[j])
  plt.axis("off")
plt.show()

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
x = layers.Conv2D(filters=32, kernel_size=6, activation="relu", padding="same")(x)
x = layers.Conv2D(filters=16, kernel_size=1, activation="relu", padding="same")(x)
outputs = layers.Conv2D(filters=3, kernel_size=3, padding="same")(x)
model = tf.keras.Model(inputs, outputs)
# Displaying the model architecture
model.summary()
    

# ========================================================================================================================================================================================================
# Training the Model
# ========================================================================================================================================================================================================
# Compile and train the model
model.compile(loss="mean_squared_error", optimizer="adam", metrics=[psnr])
history = model.fit(
    loadedPixelart, 
    loadedImages,
    validation_data=(test_x, test_y),
    batch_size=5,
    epochs=25
)
# Saving the trained model (needs to be converted to an onnx format to be compatable with Unity/Barracuda
model.save('./SRCNNModel')
print("Model Saved to: " + os.path.dirname(os.path.realpath('./SRCNNModel')) + "\SRCNNModel")


# ========================================================================================================================================================================================================
# Evaluate the model
# ========================================================================================================================================================================================================
# Creating graph to show model's accuracy & valuation accuracy per epoch    
plot3 = plt.figure(3)
plt.plot(history.history['psnr'], label='psnr')
plt.plot(history.history['val_psnr'], label = 'val_psnr')
plt.xlabel('Epoch')
plt.ylabel('psnr')
plt.legend(loc='lower right')

# Printing out final model accuracy
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print(test_acc)
# Showing graph in seperate window
plt.show()