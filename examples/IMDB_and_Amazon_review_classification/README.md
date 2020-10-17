### Running steps

Make sure you are on the `develop` branch, with `IMDB_and_Amazon_review_classification` as your working directory. 

Note: If you want to run an MNIST task, set the DATASET argument in config.yaml to "MNIST", 
otherwise if you want to run FASHION-MNIST, set the DATASET argument in config.yaml to "Fashion". 

Configure demo state
1. Create a data directory `mkdir data` and a log directory `mkdir log`
3. Modify hyper-parameters in the `config.yaml` file as needed (to set the train_epochs, batch_size, learning rate, momentum, etc)

Set up environment

4. Find and set the Alectio backend API key (found in the front end platform) with `export ALECTIO_API_KEY="KEY_HERE"`
5. Run `aws configure` to set up connection with AWS
6. Install the alectio-sdk with `pip install ../../.` (assuming you are in this current directory).
7. After activating your virtual environment (see project-level README for instructions), install all dependencies in the requirements.txt found in this repository `pip install -r requirements.txt`
8. Finally, run `python main.py --config config.yaml` to start the traning process. 

## Instructions for downloading the IMDB dataset
1. Download the IMDB dataset here: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Rename the CSV file to "imdb_reviews.csv" and place in the root directory. 

## Instructions for downloading the Amazon Reviews dataset. 
1. Because this dataset is so vast, we will only be training on a very very small subset of it (approximately 72k samples). You can change the amount we train on currently (3% of train.csv) by modifying the arg AMAZON_DATSET_TRAINING_RATIO in config.yaml
2. Download the file amazon_review_full_csv.tar.gz from this public google drive folder with many common NLP datasets. 
https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
3. Download and unzip the files. Place amazon_review_full_csv folder in this experiment folder.
4. Please note that for this task we will be grouping reviews of 4-5 as being "good", reviews of 1-2 as being "bad" and eliminating all neutral reviews (3)

## Changing the task type
- Set the `DATASET` argument in config.yaml to `"AMAZON"` to perform sentiment analysis on a subset of the amazon reviews dataset.
- Set the `DATASET` argument in config.yaml to `"IMDB"` to perform sentiment analysis on the IMDB dataset.