import argparse
import yaml, json
from alectio_sdk.flask_wrapper import Pipeline
from processes import train, test, infer, getdatasetstate

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", default="config.yaml", type=str, help="Path to config.yaml"
)

args = parser.parse_args()

with open(args.config, "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
app = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
)

if __name__ == "__main__":
    payload = json.load(open(args["sample_payload"], "r"))
    app._one_loop(args=args, payload=payload)
    # app(debug=True)
