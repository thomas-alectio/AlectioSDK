import argparse
import yaml
from alectio_sdk.flask_wrapper import Pipeline
from processes import train, test, infer, getdatasetstate

"""
parser = argparse.ArgumentParser()
parser.add_argument("--alconfig", help="Path to config.yaml",default="./config.yaml", required=False)
args = parser.parse_args()
"""

"""
def launch_app(cfgfile, configuresdk = '../../configureSDK.yaml'):
  with open(cfgfile, "r") as stream:
      args = yaml.safe_load(stream)

  with open(configuresdk, "r") as stream:
      sdkconfigs = yaml.safe_load(stream)
      
  # put the train/test/infer processes into the constructor
  AlectioPipeline = Pipeline(name=args["exp_name"],
                             train_fn=train,
                             test_fn=test,
                             infer_fn=infer,
                             getstate_fn=getdatasetstate,
                             args = args,
                             )
  app = AlectioPipeline.app
  return app
  #app.run()
  #AlectioPipeline(debug=True)
  #return AlectioPipeline

if __name__ == "__main__":
    # SAMPLE PAYLOAD
    payload = {
         "project_id": "1df477769a4211eaa6da3af9d318993f",
         "user_id": "8a90a570972811eaad5238c986352c36",
         "experiment_id": "e4cd22449a4211eaa6da3af9d318993f",
         "bucket_name": "alectio-demo",
         "cur_loop": 0,
         "type": "Object Detection",
     }
    with open("config.yaml", "r") as stream:
      args = yaml.safe_load(stream)
    
    # put the train/test/infer processes into the constructor
    app  = Pipeline(name=args["exp_name"],
                               train_fn=train,
                               test_fn=test,
                               infer_fn=infer,
                               getstate_fn=getdatasetstate,
                               args = args,
                               )
    #app = AlectioPipeline.app
    
    #app.run()
    app(debug=True)
"""
with open("./config.yaml", "r") as stream:
    args = yaml.safe_load(stream)

# put the train/test/infer processes into the constructor
AlectioPipeline = Pipeline(
    name=args["exp_name"],
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
    args=args,
)
app = AlectioPipeline.app


if __name__ == "__main__":
    # SAMPLE PAYLOAD
    payload = {
        "project_id": "1df477769a4211eaa6da3af9d318993f",
        "user_id": "8a90a570972811eaad5238c986352c36",
        "experiment_id": "e4cd22449a4211eaa6da3af9d318993f",
        "bucket_name": "alectio-demo",
        "cur_loop": 0,
        "type": "Object Detection",
    }

    # app.run()
