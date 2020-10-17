# Topic Classification on Reuters-21578 News Dataset

This example is intended to show you how to build the `train`, `test` and `infer` processes for the `AlectioSDK` for topic
classification problems. We will use the [Reuters](https://martin-thoma.com/nlp-reuters/) dataset. We focus on the 20 most popular news topics. An example sentence is



| label | topic |
| ----- | ----- |
| interest   | FED EXPECTED TO ADD RESERVES The Federal Reserve is expected to enter the U.S. government securities market to add reserves during its usual intervention period today, economists said. With federal funds trading at a steady 6-3/16 pct, most economists expect an indirect injection of temporary reserves via a medium-sized round of customer repurchase agreements. However, some economists said the Fed may arrange more aggressive system repurchase agreements. Economists would also not rule out an outright bill pass early this afternoon. Such action had been widely anticipated yesterday but failed to materialize.




### 1. Set up a virtual environment and install Alectio SDK
Before getting started, please make sure you completed the [initial installation instructions](../../README.md) to set-up your environment. 

To recap, the steps were setting up a virtual environment and then installing the AlectioSDK in that environment. 

To install the AlectioSDK from within the current directory (`./examples/NLP_classification`) run:

```
pip install ../../.
```

Next, go to the Alectio Frontend and download the API key by hitting the `DOWNLOAD API KEY` button. This will download a file called `credentials.json` which you should place in the current working directory (`./examples/NLP_classification`).

### 2. Get Code, Data and Dependencies 

First, point your terminal to the directory of this Readme file. Your terminal should look like this:
```bash 
(env)$~/AlectioSDK/examples/NLP_classification
```
Then, clone the `SDK_Reuters` branch of the topic classification repo. 
```shell
git clone --depth 1 -b SDK_Reuters --single-branch git@gitlab.com:AntonMu/reuters_hedwig.git
```
If successful, you should have a folder within your SDK repo called `reuters_hedwig`. It should look like this:

```
├── examples
│   ├── NLP_classification
│   │   └── reuters_hedwig
│   ├── image_classification
│   ├── object_detection
│   └── topic_classification
```

Then install pytorch with

```
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

```
and then the remaining requirements via:
```
pip install -r reuters_hedwig/requirements.txt
```

### 4. Build Model
We use the BERT topic classification model that is already implemented in the reuters-hedwig repo. 

### 5. Build Train, Test and Infer Processes
To use the Alectio framework we wrap the model functions of the BERT model in [`processes.py`](./processes.py).  

We can test run the processes file with:
```python
python processes.py
```

### 6. Build Flask App 
Finally, to run the flask app, execute:

```
gunicorn --bind 0.0.0.0:5000 --timeout 2500000 main:app
```
