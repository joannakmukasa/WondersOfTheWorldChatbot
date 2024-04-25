#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:24:59 2024

@author: joannamukasa
"""

# A classical NN; adapted from https://www.tensorflow.org/tutorials/keras/classification/

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# # Loading the dataset
# (train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
# output_classes = 10

# # Scale images to the [0, 1] range
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# # Make sure images have shape (28, 28, 1)
# train_images = np.expand_dims(train_images, -1)
# test_images = np.expand_dims(test_images, -1)


dataset = "archive/Wonders of World/Wonders Of World"

df_train = keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    batch_size=128,
    image_size=(256,256)
    )

df_val = keras.utils.image_dataset_from_directory(
    dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    batch_size=128,
    image_size=(256,256)
    )


output_classes = 6

class_names = df_train.class_names
print(class_names)

"""
## Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(256,256,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(output_classes)
])
"""
model= keras.Sequential(
    [
     keras.Input(shape=(256,256,3)),
     keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
     keras.layers.MaxPooling2D(pool_size=(2, 2)),
     keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
     keras.layers.MaxPooling2D(pool_size=(2, 2)),
     keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu"),
     keras.layers.MaxPooling2D(pool_size=(2, 2)),
     keras.layers.Flatten(),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(output_classes, activation="sigmoid"),
     ]
    )


model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## Train the model
# model.fit(train_images, train_labels, epochs=10, batch_size=128)

# ## Evaluate the trained model
# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\n Test accuracy:', test_acc)

# model.predict(test_images)[10]
# print(model.predict(test_images)[10])
# plt.imshow(test_images[10])


model.fit(df_train, validation_data=df_val, epochs=15, batch_size=128)

# ## Evaluate the trained model
# test_loss, test_acc = model.evaluate(df_val,  class_names, verbose=2)
# print('\n Test accuracy:', test_acc)

## Evaluate the trained model
test_loss, test_acc = model.evaluate(df_val, verbose=2)
print('\nTest accuracy:', test_acc)


# model.predict(df_val)[10]
# print(model.predict(df_val)[10])
# plt.imshow(df_val[10])

# model.save("model.h5")
# print("Saved model to disk")

