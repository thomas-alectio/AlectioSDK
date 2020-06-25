from flask import jsonify
from flask import Flask, Response
from flask import request
from flask import send_file
from waitress import serve

import numpy as np
import json
import requests
import traceback
import sys
import os
import psutil
import time
import boto3
import json
import logging
import sklearn.metrics
from copy import deepcopy

from .s3_client import S3Client
from alectio_sdk.metrics.object_detection import Metrics, batch_to_numpy
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration


class Pipeline(object):
    r"""
    A wrapper for your `train`, `test`, and `infer` function. The arguments for your functions should be specifed
    separately and passed to your pipeline object during creation.

    Args:
        name (str): experiment name
        train_fn (function): function to be executed in the train cycle of the experiment.
        test_fn (function): function to be executed in the test cycle of the experiment.
        infer_fn (function): function to be executed in the inference cycle of the experiment.
        getstate_fn (function): function specifying a mapping between indices and file names.

    """

    def __init__(self, name, train_fn, test_fn, infer_fn, getstate_fn, args):
        sentry_sdk.init(
            dsn="https://4eedcc29fa7844828397dca4afc2db32@o409542.ingest.sentry.io/5282336",
            integrations=[FlaskIntegration()]
        )
        self.app = Flask(name)
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.infer_fn = infer_fn
        self.getstate_fn = getstate_fn
        self.args = args
        self.client = S3Client()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "config.json"), "r") as f:
            self.config = json.load(f)
        # one loop
        self.app.add_url_rule("/one_loop", "one_loop", self.one_loop, methods=["POST"])
        self.app.add_url_rule("/end_exp", "end_exp", self.end_exp, methods=["POST"])

    def one_loop(self):
        # Get payload args
        payload = {
            "experiment_id": request.get_json()["experiment_id"],
            "project_id": request.get_json()["project_id"],
            "cur_loop": request.get_json()["cur_loop"],
            "user_id": request.get_json()["user_id"],
            "bucket_name": request.get_json()["bucket_name"],
            "type": request.get_json()["type"],
        }
        print('Received payload from backend')
        returned_payload = self._one_loop(payload, self.args)
        backend_ip = self.config["backend_ip"]
        port = 80
        url = "".join(["http://", backend_ip, ":{}".format(port), "/end_of_task"])
        status = requests.post(url=url, json=returned_payload, auth=('auth', os.environ['ALECTIO_API_KEY'])).status_code
        if status == 200:
            logging.info('Experiment {} running'.format(payload['experiment_id']))
            return jsonify({"Message": "Loop Started - 200 status returned"})
        else:
            return jsonify({'Message': "Loop Failed - non 200 status returned"})

    def _one_loop(self, payload, args):
        r"""
        Executes one loop of active learning. Returns the read `payload` back to the user.

        Args:
           args: a dict with the key `sample_payload` (required path) and any arguments needed by the `train`, `test`
           and infer functions.
        Example::

            args = {sample_payload: 'sample_payload.json', EXPT_DIR : "./log", exp_name: "test", \
                                                                 train_epochs: 1, batch_size: 8}
            app._one_loop(args)

        """
        # payload = json.load(open(args["sample_payload"]))
        self.logdir = payload["experiment_id"]
        if not os.path.isdir(self.logdir):
            os.mkdir(self.logdir)

        # read selected indices upto this loop
        payload['cur_loop'] = int(payload['cur_loop'])
        # self.curout_loop = payload["cur_loop"]

        # Leave cur_loop - 1 when doing a 1 dag solution, when doing 2 dag cur_loop remains the same
        self.cur_loop = payload["cur_loop"]
        self.bucket_name = payload["bucket_name"]

        # type of the ML problem
        self.type = payload["type"]

        # dir for expt log in S3
        expt_dir = [payload["user_id"], payload["project_id"], payload["experiment_id"]]

        if self.bucket_name == self.config["sandbox_bucket"]:
            # shared S3 bucket for sandbox user
            self.expt_dir = os.path.join(
                payload["user_id"], payload["project_id"], payload["experiment_id"]
            )

            self.project_dir = os.path.join(payload["user_id"], payload["project_id"])
        else:
            # dedicated S3 bucket for paid user
            self.expt_dir = os.path.join(
                payload["project_id"], payload["experiment_id"]
            )

            self.project_dir = os.path.join(payload["project_id"])

        # get meta-data of the data set
        key = os.path.join(self.project_dir, "meta.json")
        bucket = boto3.resource("s3").Bucket(self.bucket_name)
        json_load_s3 = lambda f: json.load(bucket.Object(key=f).get()["Body"])
        self.meta_data = json_load_s3(key)

        # self.meta_data = self.client.read(self.bucket_name, key, "json")
        logging.info('SDK Retrieved file: {} from bucket : {}'.format(key, self.bucket_name))

        if self.cur_loop == 0:
            self.resume_from = None
            self.state_json = self.getstate_fn(args)
            object_key = os.path.join(self.expt_dir, "data_map.pkl")
            self.client.multi_part_upload_with_s3(self.state_json, self.bucket_name, object_key, "pickle")
        else:
            # two dag approach needs to refer to the previous checkpoint
            self.resume_from = "ckpt_{}".format(self.cur_loop - 1)

        self.ckpt_file = "ckpt_{}".format(self.cur_loop)

        self.train(args)
        self.test(args)
        self.infer(args)

        # Drop unwanted payload values
        del payload["type"]
        del payload["cur_loop"]
        del payload["bucket_name"]
        return payload

    def train(self, args):
        r"""
        A wrapper for your `train` function. Returns `None`.

        Args:
           args: a dict whose keys include all of the arguments needed for your `train` function which is defined in `processes.py`.

        """
        start = time.time()

        self.labeled = []
        for i in range(self.cur_loop + 1):
            object_key = os.path.join(
                self.expt_dir, "selected_indices_{}.pkl".format(i)
            )
            selected_indices = self.client.read(
                self.bucket_name, object_key=object_key, file_format="pickle"
            )
            self.labeled.extend(selected_indices)
        self.labeled.sort()  # Maintain increasing order
        labels = self.train_fn(args,
                               labeled=deepcopy(self.labeled),
                               resume_from=self.resume_from,
                               ckpt_file=self.ckpt_file,
                               )

        end = time.time()

        # @TODO compute insights from labels
        insights = {"train_time": end - start}
        object_key = os.path.join(
            self.expt_dir, "insights_{}.pkl".format(self.cur_loop)
        )

        self.client.multi_part_upload_with_s3(insights, self.bucket_name, object_key, "pickle")

        return

    def test(self, args):
        r"""
        A wrapper for your `test` function which writes predictions and ground truth to the specified S3 bucket. Returns `None`.

        Args:
           args: a dict whose keys include all of the arguments needed for your `test` function which is defined in `processes.py`.

        """
        res = self.test_fn(args, ckpt_file=self.ckpt_file)

        predictions, ground_truth = res["predictions"], res["labels"]

        # write predictions and labels to S3
        object_key = os.path.join(
            self.expt_dir, "test_predictions_{}.pkl".format(self.cur_loop)
        )
        self.client.multi_part_upload_with_s3(predictions, self.bucket_name, object_key, "pickle")

        if self.cur_loop == 0:
            # write ground truth to S3
            object_key = os.path.join(
                self.expt_dir, "test_ground_truth.pkl".format(self.cur_loop)
            )
            self.client.multi_part_upload_with_s3(ground_truth, self.bucket_name, object_key, "pickle")

        self.compute_metrics(predictions, ground_truth)
        return

    def compute_metrics(self, predictions, ground_truth):
        if self.type == "Object Detection":
            det_boxes, det_labels, det_scores, true_boxes, true_labels = batch_to_numpy(
                predictions, ground_truth
            )

            m = Metrics(
                det_boxes=det_boxes,
                det_labels=det_labels,
                det_scores=det_scores,
                true_boxes=true_boxes,
                true_labels=true_labels,
                num_classes=len(self.meta_data["class_labels"]),
            )

            metrics = {
                "mAP": m.getmAP(),
                "AP": m.getAP(),
                "precision": m.getprecision(),
                "recall": m.getrecall(),
                "confusion_matrix": m.getCM().tolist(),
                "class_labels": self.meta_data["class_labels"],
            }

        if self.type == "Classification" or self.type == "Text Classification":
            confusion_matrix = sklearn.metrics.confusion_matrix(
                ground_truth, predictions
            )
            num_queried_per_class = {
                k: v for k, v in enumerate(confusion_matrix.sum(axis=1))
            }
            acc_per_class = {
                k: v.round(3)
                for k, v in enumerate(
                    confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
                )
            }
            accuracy = sklearn.metrics.accuracy_score(ground_truth, predictions)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = confusion_matrix.diagonal()
            TN = confusion_matrix.sum() - (FP + FN + TP)
            label_disagreement = {k: v.round(3) for k, v in enumerate(FP / (FP + TN))}

            metrics = {
                "accuracy": accuracy,
                "confusion_matrix": confusion_matrix,
                "acc_per_class": acc_per_class,
                "label_disagreement": label_disagreement,
                "num_queried_per_class": num_queried_per_class,
            }

        # save metrics to S3
        object_key = os.path.join(
            self.expt_dir, "metrics_{}.pkl".format(self.cur_loop)
        )
        self.client.multi_part_upload_with_s3(metrics, self.bucket_name, object_key, "pickle")
        return

    def infer(self, args):
        r"""
        A wrapper for your `infer` function which writes outputs to the specified S3 bucket. Returns `None`.

        Args:
           args: a dict whose keys include all of the arguments needed for your `infer` function which is defined in `processes.py`.

        """
        ts = range(self.meta_data["train_size"])
        self.unlabeled = sorted(list(set(ts) - set(self.labeled)))
        outputs = self.infer_fn(args,
                                unlabeled=deepcopy(self.unlabeled), ckpt_file=self.ckpt_file
                                )["outputs"]

        # Remap to absolute indices
        remap_outputs = {}
        for i, (k, v) in enumerate(outputs.items()):
            ix = self.unlabeled.pop(0)
            remap_outputs[ix] = v

        # write the output to S3
        key = os.path.join(
            self.expt_dir, "infer_outputs_{}.pkl".format(self.cur_loop)
        )
        self.client.multi_part_upload_with_s3(remap_outputs, self.bucket_name, key, "pickle")
        return

    def __call__(self, debug=False, host="0.0.0.0", port=5000):
        r"""
        A wrapper for your `test` function which writes predictions and ground truth to the specified S3 bucket. Returns `None`.

        Args:
           debug (boolean, Default=False): If set to true, then the app runs in debug mode. See https://flask.palletsprojects.com/en/1.1.x/api/#flask.Flask.debug.
           host (str, Default='0.0.0.0'): the hostname to be listened to.
           port(int, Default:5000): the port of the webserver.

        """
        serve(self.app, host="0.0.0.0", port=5000)

    @staticmethod
    def shutdown_server():
        func = request.environ.get('werkzeug.server.shutdown')
        if func is None:
            raise RuntimeError('Not running with the Werkzeug Server')
        func()

    @staticmethod
    def end_exp():
        print()
        print('======== Experiment Ended ========')
        print('Server shutting down...')
        p = psutil.Process(os.getpid())
        p.terminate()
