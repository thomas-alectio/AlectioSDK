from alectio_sdk.flask_wrapper import Pipeline
from processes import train, test, infer, getdatasetstate

# put the train/test/infer processes into the constructor
app = Pipeline(
    name="coco",
    train_fn=train,
    test_fn=test,
    infer_fn=infer,
    getstate_fn=getdatasetstate,
)

if __name__ == "__main__":

    # SAMPLE PAYLOAD
    # payload = {
    #     "project_id": "1df477769a4211eaa6da3af9d318993f",
    #     "user_id": "8a90a570972811eaad5238c986352c36",
    #     "experiment_id": "e4cd22449a4211eaa6da3af9d318993f",
    #     "bucket_name": "alectio-demo",
    #     "cur_loop": 0,
    #     "type": "Object Detection",
    # }

    # app._one_loop(payload)
    
    app(debug=True)
