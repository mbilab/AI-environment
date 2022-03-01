#!/usr/local/bin/python3
import sys

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def getMNISTData():
    print("Preparing data ...")
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = (x_train[:1000], y_train[:1000]), (
        x_test[:100],
        y_test[:100],
    )

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples\n")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train[:1000], y_train[:1000]), (x_test[:100], y_test[:100])


def getModel():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    print("")
    model.summary()
    print("")
    return model


def run(model, train_data, test_data, batch_size, n_epochs):
    print(f"Start training ...\n")
    x_train, y_train = train_data
    x_test, y_test = test_data
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    model.fit(
        x_train, y_train, batch_size=batch_size, epochs=n_epochs, validation_split=0.1
    )

    print(f"\nStart testing ...")
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])


if __name__ == "__main__":
    batch_size = 128
    n_epochs = 10
    train_data, test_data = getMNISTData()
    model = getModel()
    run(model, train_data, test_data, batch_size, n_epochs)
