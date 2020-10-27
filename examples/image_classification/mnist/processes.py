import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import *
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import warnings
import tarfile
import fnmatch
import os.path
import shutil
from pathlib import Path
import joblib

tf.executing_eagerly()
tf.config.set_soft_device_placement(True)
warnings.filterwarnings("ignore")

##### Global variables
mnist = tf.keras.datasets.mnist
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


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


def make_tarfile(output_filename, filenames):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(filenames[0], arcname=os.path.basename(filenames[0]))


def tar_from_checkpoint(args, ckpt_file):
    '''
    files_to_compress = []
    for file in os.listdir(args['EXPT_DIR']):
        if fnmatch.fnmatch(file, ckpt_file + '*'):
            files_to_compress.append((args['EXPT_DIR'], file))
    '''

    make_tarfile(os.path.join(args['EXPT_DIR'], ckpt_file + '.tar.gz'), os.path.join(args['EXPT_DIR'], ckpt_file))
    # shutil.rmtree(os.path.join(args['EXPT_DIR'], ckpt_file))

    """
    for path_to_file, file in files_to_compress:
        if os.path.join(path_to_file, file) != os.path.join(args['EXPT_DIR'], ckpt_file + '.tar.gz'):
            os.remove(os.path.join(path_to_file, file))
    """


def train(args, ckpt_file, labeled, resume_from):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_train = x_train[labeled, ...]
    y_train = y_train[labeled]

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(args['BATCH_SIZE'])

    optimizer = tf.keras.optimizers.Adam()
    model = MyModel()

    # load from checkpoint if resume_from is not None
    if resume_from is not None:
        file_name = os.path.basename(resume_from)
        index_of_dot = file_name.index('.')
        ckpt_file_no_ext = file_name[:index_of_dot]
        resume_from_path = os.path.join(args['EXPT_DIR'], ckpt_file_no_ext, '.')
        model.load_weights(resume_from_path)

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

    train_epochs = args['EPOCHS']
    average_loss = []
    average_acc = []
    total_labels = []
    total_predictions = []

    for epoch in range(train_epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        for images, labels in train_ds:
            label, prediction = train_step(images, labels)
            total_labels.extend(label.numpy())
            for p in prediction:
                predictions = tf.nn.softmax(p).numpy()
                total_predictions.append(np.argmax(predictions, axis=0))
            # average_loss.append(train_loss.result())
            # average_acc.append(train_accuracy.result() * 100)

        if epoch % args["DISPLAY_STEP"] == 0:
            average_loss = np.mean(average_loss)
            average_acc = np.mean(average_acc)
            print(
                f'Epoch {epoch + 1}, '
                f'Train Loss: {train_loss.result()}, '
                f'Train Accuracy: {train_accuracy.result() * 100}, '
            )
            average_acc = []
            average_loss = []

    print("Finished Training. Saving the model as {}".format(ckpt_file))

    file_name = os.path.basename(ckpt_file)
    index_of_dot = file_name.index('.')
    ckpt_file_no_ext = file_name[:index_of_dot]
    checkpoint_path = os.path.join(args['EXPT_DIR'], ckpt_file_no_ext)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    else:
        shutil.rmtree(checkpoint_path)
        os.mkdir(checkpoint_path)

    model.save_weights(os.path.join(checkpoint_path, '.'), save_format='tf')
    # a method which takes the three tf checkpoint files generated and tars them into one compressed tarball
    tar_from_checkpoint(args, ckpt_file_no_ext)

    return {'labels': total_labels, 'predictions': total_predictions}


def test(args, ckpt_file):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_test = x_test[..., tf.newaxis].astype("float32")

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    print("Loading latest checkpoint")
    model = MyModel()

    file_name = os.path.basename(ckpt_file)
    index_of_dot = file_name.index('.')
    ckpt_file_no_ext = file_name[:index_of_dot]

    model.load_weights(os.path.join(args['EXPT_DIR'], ckpt_file_no_ext, '.'))

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    total_labels = []
    total_predictions = []

    average_loss = []
    average_acc = []

    @tf.function
    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        return labels, predictions

    for test_images, test_labels in test_ds:
        label, prediction = test_step(test_images, test_labels)
        total_labels.extend(label.numpy())
        for p in prediction:
            predictions = tf.nn.softmax(p).numpy()
            total_predictions.append(np.argmax(predictions, axis=0))
        average_loss.append(test_loss.result())
        average_acc.append(test_accuracy.result() * 100)

    average_loss = np.mean(average_loss)
    average_acc = np.mean(average_acc)

    print(f'Test Loss: {average_loss}' f'Test Accuracy: {average_acc}')

    return {
        "predictions": total_predictions,
        "labels": total_labels
    }


def infer(args, unlabeled, ckpt_file):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    unlabeled_images = x_train[..., tf.newaxis].astype("float32")
    unlabeled_images = unlabeled_images[unlabeled, ...]

    # Load the last best model
    print("Loading latest checkpoint")
    model = MyModel()
    file_name = os.path.basename(ckpt_file)
    index_of_dot = file_name.index('.')
    ckpt_file_no_ext = file_name[:index_of_dot]
    model.load_weights(os.path.join(args['EXPT_DIR'], ckpt_file_no_ext, '.'))
    outputs = {}

    # Get raw output from model
    presoft = model(unlabeled_images)
    for index, output in enumerate(presoft):
        softmax = tf.nn.softmax(output).numpy()
        prediction = np.argmax(softmax, axis=0)
        outputs[index] = {}
        outputs[index]["prediction"] = prediction
        outputs[index]["pre_softmax"] = output.numpy()

    return {"outputs": outputs}


def getdatasetstate(args, split="train"):
    columns = ["label"] + list(range(args["NUM_INPUT"]))
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
        pathdict[ix] = str(row["label"]) + "_label.jpg"

    return None


if __name__ == "__main__":
    with open("./config.yaml", "r") as stream:
        args = yaml.safe_load(stream)
    labeled = list(range(20000))
    resume_from = None
    ckpt_file = "ckpt_0.tar.gz"
    train(args, labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(args, ckpt_file=ckpt_file)
    # infer(args, unlabeled=[range(1000, 2000)], ckpt_file=ckpt_file)
    # ret = getdatasetstate(args)

    print('Done with train, test, infer')
