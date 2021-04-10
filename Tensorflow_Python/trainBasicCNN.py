import tensorflow as tf
import tensorflow_datasets as tfds
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


# Data Augmentation
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(32, 32),
  layers.experimental.preprocessing.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal")
  #layers.experimental.preprocessing.RandomRotation(0.2),
])


# Creating neural network
inputs = tf.keras.Input(shape=(32, 32, 3))
# Data Augmentation
x = data_augmentation(inputs)
# Convolutional Base
x = layers.Conv2D(32, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Dropout(0.2)(x)
# Dense Layers
x = layers.Flatten()(x)
x = layers.Dense(8)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(8)(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Compile and train the model
model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=5000, epochs=25, validation_data=(test_images, test_labels))


# Displaying the model architecture
model.summary()
model.save('./TestModel')

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