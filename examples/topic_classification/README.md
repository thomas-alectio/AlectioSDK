# Topic Classification on DailyDialog Dataset

This example is intended to show you how to build the `train`, `test` and `infer` processes for the `AlectioSDK` for topic
classification problems. We will use the [DailyDialog](https://arxiv.org/abs/1710.03957) dataset. Each sample in this
dataset is a conversation between two people. The objective is to classify the topic of their conversation. The topics are labeled as following:

| label | topic |
| ----- | ----- |
| 0    | Ordinary Life | 
| 1     | School Life | 
| 2    | Culture & Education | 
| 3    | Attitude & Emotion | 
| 4    | Relationship |
| 5     | Tourism | 
| 6    | Health | 
| 7    | Work |
| 8     | Politics | 
| 9     | Finance | 

Since the size of this dataset is small, it is included in this repo in the `./data` directory. 


*** All of the following steps assume that your terminal points to the current directory, i.e. `./examples/topic_classification` *** 

### 1. Set up a virtual environment and install Alection SDK
Before getting started, please make sure you completed the [initial installation instructions](../../README.md) to set-up your environment. 

To recap, the steps were setting up a virtual environment and then installing the AlectioSDK in that environment. 

To install the AlectioSDK from within the current directory (`./examples/topic_classification`) run:

```
pip install ../../.
```
### 2. Get Data and Dependencies 

Install the requirements via:
```
pip install -r requirements.txt
```

We use `spacy` to parse the text data. The module was already
installed as part of the requirements, but we still need to download the English 
model for it via
```
python -m spacy download en
```

#### Download GloVe vectors
We will use GloVe 6B.100d for word embedding. 

We download the vectors by running:
```
mkdir vector
cd vector
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ..
```
After you unzipped it, the directory structure of 
`vector` should look like this:
```
├── vector 
    ├── glove.6B.50d.txt
    ├── glove.6B.100d.txt
    ├── glove.6B.200d.txt
    └── glove.6B.300d.txt
```
In this example, we will only need `glove.6B.100d.txt`.

#### Create a log directory
Create a log directory in the project root to save checkpoints
```
mkdir log
```
#### Set Environment Variables 
To set the environment variables, we run

```
source setenv.sh 
```
The variables are as follows:

| variable | meaning | 
| -------- | ------- |
| VECTOR_DIR | directory where word vectors are saved |
| EXPT_DIR | directory where experiment log and checkpoints are saved |
| DATA_DIR | directory where processed data is saved | 
| DEVICE   | cpu or gpu device where model training/testing/inference takes place | 
| FLASK_ENV | the environment for the Flask app we build |

### 3. Build Dataset object
We will use `pytorch` and `torchtext` for this project. We build a dataset
object `DailyDialog`, text and label fields there. Please checkout the code
in [`dataset.py`](./dataset.py) for more details.

### 4. Build Model
We will use a 2-layer bidirectional LSTM for text classification. For
the architecture of the model see [`model.py`](./model.py).

### 5. Build Train, Test and Infer Processes
The train, test and infer processes are implemented in [`processes.py`](./processes.py).  
For more information on each, refer to the doc strings of each function.


Run this step via
```python
python processes.py
```

### 6. Build Flask App 
Finally, to run the flask app, execute:

```python
python main.py --expname <experiment-name>
```
## Return Types

The return from the `test` and `infer` will be sent to Alectio's platform for 
computing the model performance and making active learning decisions. 
Therefore, it is important to make sure that they have the expected format.

#### Format of the Test Outputs
Return from the `test` process will be used to compute performance metrics of
your model. The return is a dictionary with two keys

| key | value |
| --- | ----- | 
| predictions | a dictionary of test data index and model's prediction |
| labels | a dictionary of test data index and its ground-truth label | 

For classification problem, the model's prediction and 
the ground-truth of each test sample is an integer indicating the class label.

For example, if the test set consists of `(n + 1)` samples indexed by `(0, 1, ..., n)`,
then the value of `predictions` looks like
```python
{
    0: x0,
    1: x1,
    ...
    n: xn
}
```
where `xi` is the integer class label of sample `i` predicted by the model. 

The value of `labels` looks like
```python
{
    0: y0,
    1: y1,
    ...
    n: yn
}
```
where `yi` is the integer ground-truth class label of sample `i`.

<!-- The infer process is missing here -->
