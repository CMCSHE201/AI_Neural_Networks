import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter


# Download and prepare the CIFAR10 dataset
# Downloading dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize dataset pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0


# Building the Neural Network
# Create the convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add Dense layers on top
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# Displaying the model architecture
model.summary()


# Compile and train the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


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
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
# Showing graph in seperate window
plt.show()