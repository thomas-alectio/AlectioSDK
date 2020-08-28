import argparse
import yaml, json
from alectio_sdk.flask_wrapper import Pipeline
from processes import train, test, infer, getdatasetstate

with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
AlectioPipeline  = Pipeline(
                            name=args["exp_name"],
                            train_fn=train,
                            test_fn=test,
                            infer_fn=infer,
                            getstate_fn=getdatasetstate,
                            args=args,
)

app = AlectioPipeline.app

if __name__ == "__main__":
    # payload = json.load(open(args["sample_payload"], "r"))
    # app._one_loop(args=args, payload=payload)
    app.run()
