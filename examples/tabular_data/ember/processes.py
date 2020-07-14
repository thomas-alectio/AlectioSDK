import time
import os
import random
import json
import sys
import lightgbm as lgb
import pandas as pd
import numpy as np
import ember
from sklearn.externals import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit


# Get env variables
CKPT_FILE = os.getenv('CKPT_FILE')
DEVICE = os.getenv('CUDA_DEVICE')
TASK = os.getenv('TASK')
RESUME_FROM = os.getenv('RESUME_FROM')
EXPT_DIR = os.getenv('EXPT_DIR')
DATA_DIR = os.getenv('DATA_DIR')


def preprocess_data():
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in split.split(X_train, y_train):
        X_train, X_test = X_train[train_index], X_train[test_index]
        y_train, y_test = y_train[train_index], y_train[test_index]
    X_orig_train = pd.DataFrame(X_train)
    X_orig_test = pd.DataFrame(X_test)
    y_orig_train = pd.DataFrame(y_train)
    y_orig_test = pd.DataFrame(y_test)
    # train = 20000 samples and test = 5000 samples
    X_train = X_orig_train[:20000]
    X_test = X_orig_test[:5000]
    y_train = pd.Series(y_orig_train[:20000].values.ravel())
    y_test = pd.Series(y_orig_test[:5000].values.ravel())
    return {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}


def train(args, labeled, resume_from, ckpt_file):
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(args["DATA_DIR"])
    model = LGBMClassifier(objective='binary', n_jobs=-1)
    if RESUME_FROM:
        resume_from = os.path.join(args['EXPT_DIR'], resume_from)
        model = joblib.load(resume_from)
    model.fit(X_train.loc[labeled, :], y_train[labeled])
    # save model
    ckpt_path = os.path.join(args['EXPT_DIR'], ckpt_file)
    joblib.dump(model, ckpt_path)
    print("Finished Training. Saving the model as {}".format(ckpt_path))
    return


def test(args, ckpt_file):
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(args["DATA_DIR"])
    ckpt_path = os.path.join(args['EXPT_DIR'], ckpt_file)
    model = joblib.load(ckpt_path)
    prediction = model.predict(X_test)
    # save prediction
    with open(os.path.join(EXPT_DIR, 'prediction.pkl'), 'wb') as f:
        pickle.dump(prediction, f)
    return {"predictions": predictions, "labels": targets}


def infer(args, unlabeled, ckpt_file):
    # load unlabeled data
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(args["DATA_DIR"])
    # load ckpt trained in the current loop
    ckptpath = os.path.join(args['EXPT_DIR'], ckpt_file)
    model = joblib.load(ckptpath)
    output = {}
    for rec in unlabeled:
        output[rec] = {}
        output[rec]['softmax'] = model.predict_proba(rec)
        output[rec]['prediction'] = model.predict(X_train[rec])

    return {"outputs": outputs}


def getdatasetstate():
    return {k: k for k in range(20000)}
