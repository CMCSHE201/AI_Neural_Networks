import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from data_generator import train_data_generator, test_data_generator
from utils import psnr

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter

DATA_DIR = "./src/"
TRAIN_PATH = "DIV2K_train_HR"
TEST_PATH = "DIV2K_valid_HR"

N_TRAIN_DATA = 800
N_TEST_DATA = 100
BATCH_SIZE = 50
EPOCHS = 10


# ========================================================================================================================================================================================================
# Defining a specific layer for handling data augmentation
# ========================================================================================================================================================================================================
train_data_generator = train_data_generator(
    DATA_DIR, TRAIN_PATH, scale=4.0, batch_size=BATCH_SIZE
)

test_x, test_y = next(
    test_data_generator(
        DATA_DIR, TEST_PATH, scale=4.0, batch_size=N_TEST_DATA, shuffle=False
    )
)


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
x = layers.Conv2D(filters=64, kernel_size=9, activation="relu", padding="same", input_shape=(None,None,3),)(x)
x = layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding="same")(x)
outputs = layers.Conv2D(filters=3, kernel_size=5, padding="same")(x)
model = tf.keras.Model(inputs, outputs)
# Displaying the model architecture
model.summary()
    

# ========================================================================================================================================================================================================
# Training the Model
# ========================================================================================================================================================================================================
# Compile and train the model
model.compile(loss="mean_squared_error", optimizer="adam", metrics=[psnr])
history = model.fit(
    train_data_generator,
    validation_data=(test_x, test_y),
    steps_per_epoch=N_TRAIN_DATA // BATCH_SIZE,
    epochs=EPOCHS
)
# Saving the trained model (needs to be converted to an onnx format to be compatable with Unity/Barracuda
model.save('./SRCNNModel')


# ========================================================================================================================================================================================================
# Evaluate the model
# ========================================================================================================================================================================================================
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