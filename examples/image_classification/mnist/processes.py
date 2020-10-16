# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:52:22 2020

@author: arun
"""
import os
import yaml
import pandas as pd
import tensorflow as tf
from utils import *
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow.experimental.numpy as tnp


class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

##### Global variables
mnist = tf.keras.datasets.mnist
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = MyModel()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

def train(args, ckpt_file):
    optimizer = tf.keras.optimizers.Adam(learning_rate=args['INITIAL_LR'])
    checkpoint_prefix = os.path.join(args["OUTPUT_DIRECTORY"], ckpt_file)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(args['BATCH_SIZE'])

    best_acc = 0.0
    EPOCHS = args['EPOCHS']
    average_loss = 0.0
    average_acc = 0.0

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)
        return labels, predictions

    total_labels = []
    total_predictions = []

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        for images, labels in train_ds:
            label, prediction = train_step(images, labels)
            total_labels.append(tnp.asarray(label))
            total_predictions.append(tnp.asarray(prediction))
            average_loss += train_loss.result()
            average_acc += train_accuracy.result() * 100

        if epoch % args["DISPLAY_STEP"] == 0:
            if epoch > 0:
                average_loss /= args["DISPLAY_STEP"]
                average_acc /= args["DISPLAY_STEP"]
            print(
                f'Epoch {epoch + 1}, '
                f'Train Loss: {average_loss}, '
                f'Train Accuracy: {average_acc}, '
            )
            average_acc = 0.0
            average_loss = 0.0

        if average_acc > best_acc:
            best_acc = average_acc
            checkpoint.save(file_prefix=checkpoint_prefix)

    print('Completed Training...')
    return {'labels': total_labels, 'predictions': total_predictions}


def test(args, ckpt_file):
    checkpoint = tf.train.Checkpoint(model=NN, optimizer=optimizer)

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    print("Loading latest checkpoint")
    checkpoint.restore(tf.train.latest_checkpoint(args["OUTPUT_DIRECTORY"]))

    print("Testset Accuracy: {:.4f}".format(test_acc))
    y_pred, y_true = getpreds(NN, testX, testY)

    return {
        "predictions": y_pred.numpy().reshape(len(y_pred), 1),
        "labels": y_true.numpy().reshape(len(y_true), 1),
    }


def infer(args, unlabeled, ckpt_file):
    optimizer = tf.optimizers.Adam(learning_rate=args["INITIAL_LR"])
    checkpoint_prefix = os.path.join(args["OUTPUT_DIRECTORY"], "ckpt_0")
    checkpoint = tf.train.Checkpoint(model=NN, optimizer=optimizer)

    # Load the last best model
    print("Loading latest checkpoint")
    checkpoint.restore(tf.train.latest_checkpoint(args["OUTPUT_DIRECTORY"]))
    unlabeled_images = mnist.train.images[unlabeled, ...]
    unlabeled_classes = mnist.train.labels[unlabeled, ...]
    presoft = NN(unlabeled_images)
    presoftarr = presoft.numpy().reshape(len(presoft), 1)

    predictions = {i: row for i, row in enumerate(presoftarr)}

    return {"outputs": predictions}


def getdatasetstate(args, split="train"):
    columns = ["Label"] + list(range(args["NUM_INPUT"]))
    if split == "train":
        train_label_file = "{}mnist_train.csv".format(args["LABEL_DIRECTORY"])
        dataframe = pd.read_csv(train_label_file)
        dataframe.columns = columns
    else:
        test_label_file = "{}mnist_test.csv".format(args["LABEL_DIRECTORY"])
        dataframe = pd.read_csv(test_label_file)
        dataframe.columns = columns

    pathdict = {}
    for ix, row in dataframe.iterrows():
        pathdict[ix] = row[
            "Label"
        ]  ######### here index carries the necessary reference

    return pathdict


if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)

