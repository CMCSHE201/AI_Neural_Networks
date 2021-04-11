import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter


# ========================================================================================================================================================================================================
# Creating network datasets
# ========================================================================================================================================================================================================
# Loading images from directorys
def loadImageSet(
    data_dir, mode, target_size=(1024, 1024), batch_size=5, shuffle=False
):
    datagen = ImageDataGenerator()
    gen = datagen.flow_from_directory(
        directory=data_dir,
        classes=[mode],
        class_mode=None,
        color_mode="rgb",
        target_size=target_size,
        batch_size=batch_size,
        shuffle=shuffle)
    for imgs in gen:
        yield imgs / 255.0

print(os.listdir("./images_pixel"))
loadedPixelart = next(
    loadImageSet(
        "./", "images_pixel", target_size=(128,128), batch_size=291, shuffle=False)
    )
print(os.listdir("./images_normal"))
loadedImages = next(
    loadImageSet(
        "./", "images_normal", target_size=(1024,1024), batch_size=291, shuffle=False)
    )
# Splitting loaded image sets into training datasets and test datasets
train_x, test_x, train_y, test_y = train_test_split( np.array(loadedPixelart) , np.array(loadedImages) , test_size=0.1 )
# Showing first 9 elements in each image set to show that the images are indexed/ accossiated properly
plt.figure(figsize=(10, 10))
for i in range(4):
  ax = plt.subplot(2, 2, i + 1)
  plt.imshow(train_x[i])
  plt.axis("off")
plt.figure(figsize=(10, 10))
for j in range(4):
  ax = plt.subplot(2, 2, j + 1)
  plt.imshow(train_y[j])
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
inputs = tf.keras.Input(shape=(128, 128, 3))
# Data Augmentation
x = data_augmentation(inputs)
# Convolutional Base
x = layers.UpSampling2D((2,2))(x)
x = layers.UpSampling2D((2,2))(x)
x = layers.UpSampling2D((2,2))(x)
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
model.compile(loss="mean_squared_error", optimizer="adam", metrics=['mean_squared_error', 'accuracy'])
history = model.fit(
    loadedPixelart, 
    loadedImages,
    validation_data=(test_x, test_y),
    batch_size=1,
    epochs=10
)
# Saving the trained model (needs to be converted to an onnx format to be compatable with Unity/Barracuda
model.save('./SRCNNModel')
print("Model Saved to: " + os.path.dirname(os.path.realpath('./SRCNNModel')) + "\SRCNNModel")


# ========================================================================================================================================================================================================
# Evaluate the model
# ========================================================================================================================================================================================================
# Creating graph to show model's accuracy & valuation accuracy per epoch    
plot3 = plt.figure(3)
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

plot4 = plt.figure(4)
plt.plot(history.history['mean_squared_error'], label='mean_squared_error')
plt.plot(history.history['val_mean_squared_error'], label = 'val_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('mean_squared_error')
plt.legend(loc='lower right')

# Printing out final model accuracy
#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#print(test_acc)
# Showing graph in seperate window
plt.show()