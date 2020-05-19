from alectio_sdk.flask_wrapper import Pipeline
from processes import train, test, infer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--expname",
    type=str,
    default="ActiveLearningexperiment",
    help="A name of your choice for your flask server",
)

args = parser.parse_args()

# put the train/test/infer processes into the constructor
app = Pipeline(name=args.expname, train_fn=train, test_fn=test, infer_fn=infer)


if __name__ == "__main__":
    app(debug=True)
