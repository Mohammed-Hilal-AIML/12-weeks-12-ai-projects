# Download and extract dataset

### Steps to prepare the dataset:
  - Goto [Sentiment140 dataset with 1.6 million tweets](https://www.kaggle.com/kazanova/sentiment140) and download the dataset
  - Extract the dataset and put the `.csv` file in the `/data` directory
  - Rename the file as `sentiment.csv`
  - Run `tokenizer.ipynb` cell by cell to generate the tokenized version of the dataset

### Training
1. Successful tokenization will generate 6 files:
    - input_ids_train.pt
    - attention_mask_train.pt
    - y_train.pt
    - input_ids_test.pt
    - attention_mask_test.pt
    - y_tests.pt
2. Create virtualenv using requirements.txt, run `sentiment_analysis.ipynb` cell by cell to train the model.
3. Models training checkpoints are saved at the end of every epoch as `DistilBERT_epoch_<EPOCH_NUMBER>.pt`

### FAQs
  - This sentiment analysis is modeled based on Huggingface Transformers DistilBert model. For more info on DistilBert click [here](https://huggingface.co/transformers/model_doc/distilbert.html).
  - The dataset was tokenized in advanced to reduce memory usage. This might not be an issue for everyone. So feel free to skip this part.
  - The Sentiment140 dataset is a huge (1.6 Million) collection of tweets and has only two labeled sentiments, positive and negative.