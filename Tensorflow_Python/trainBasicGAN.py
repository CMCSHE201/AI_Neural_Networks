import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

import os

import tensorflow as tf
from tensorflow.keras import datasets, layers, models


url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

data_path = tf.keras.utils.get_file(origin=url, untar=True, fname="BSR") # Downloads and extracts the filesystem
root_path = os.path.join(data_path, "BSDS500/data")   

target_size = 300
upscale_factor = 3
input_size = target_size // upscale_factor

# Creates the Trainig and Validation datasets and sets image sizes to 256x256 by default
train_data_raw = tf.keras.preprocessing.image_dataset_from_directory(root_path, label_mode=None, image_size=(target_size, target_size),validation_split=0.2, subset="training",   seed=42)
validate_data_raw = tf.keras.preprocessing.image_dataset_from_directory(root_path, label_mode=None, image_size=(target_size, target_size),validation_split=0.2, subset="validation", seed=42)


# Scales the images so that each pixel value ranges from 0 to 1
def scale(img):
    return (img)/255.0

train_data = train_data_raw.map(scale)
validate_data = validate_data_raw.map(scale)

def process_input(img, target_size, upscale_factor):
    return tf.image.resize(img, [target_size, target_size], method="area")


def process_target(img):
    img = tf.image.rgb_to_yuv(img) # Change the image format to yuv scale, 
    return tf.split(img, 3, axis=3)[0]


train_data_yuv = train_data.map(process_target)
train_data_scaled = train_data_yuv.map(lambda img: (process_input(img, input_size, upscale_factor), img))

train_ds = train_data_scaled.prefetch(buffer_size=32) #

validate_data_yuv = validate_data.map(process_target)
validate_data_scaled = validate_data_yuv.map(lambda img: (process_input(img, input_size, upscale_factor), img))

valid_ds = validate_data_scaled.prefetch(buffer_size=32) #



loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
epochs = 250

#Defining the network with Convolutional layers
conv_args = {"activation": "relu",
             "kernel_initializer": "Orthogonal",
             "padding": "same"}

inputs = tf.keras.Input(shape=(100, 100, 1))
# Convolutional Base
x = layers.Conv2D(64, 5, **conv_args)(inputs)
x = layers.Conv2D(64, 5, **conv_args)(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(32, 3, **conv_args)(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(32, 3, **conv_args)(x)
x = layers.Conv2D(1 * (upscale_factor ** 2), 3, **conv_args)(x)

# Dense Layers
x = layers.Flatten()(x)
x = layers.Dense(5625)(x)
x = layers.Activation('relu')(x)
x = layers.Dense(5625)(x)
x = layers.Activation('relu')(x)
x = layers.Reshape((25,25,9))(x)

# Upsampling
x = layers.UpSampling2D((2, 2))(x)
x = layers.UpSampling2D((2, 2))(x)

## Setting up model output
x = layers.Reshape([100, 100, 9])(x)
outputs = tf.nn.depth_to_space(x, upscale_factor)

#Compiling the model
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=optimizer,
    loss=loss_function, 
    metrics=['accuracy'])

#Training the model
history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, verbose=2)


# Displaying the model architecture
model.summary()


# Evaluate the model
# Creating graph to show model's accuracy & valuation accuracy per epoch
plot1 = plt.figure(1)
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