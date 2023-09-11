
import os 
import sys
sys.path.insert(0, '..')
from pathlib import Path
import pandas as pd
from prefect import task, flow, get_run_logger, tasks
from typing import List, Any, Tuple
import datetime
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
import logging
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import string
import nltk # Natural Language tool kit
nltk.download('stopwords')
# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords

from store.read_write_s3  import read_s3_file, upload_to_aws

with initialize(version_base=None, config_path="../config"):
    config = compose(config_name="main", overrides=["etl=etl1"])    
    # print(OmegaConf.to_yaml(config))


stopwords.words(config.etl.stop_words)   


@task(retries=3, retry_delay_seconds=5)
def read_data() -> pd.DataFrame:
    # url = "https://min-api.cryptocompare.com/data/pricemulti?fsyms=BTC,ETH,REP,DASH&tsyms=USD"
    ''' Load data from path'''
    file_path = Path(config.etl.data.raw)
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".tsv":
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(
            "File format not supported. Please use a CSV or TSV file.")
    return df


def message_cleaning(message: string) -> List:
    '''# clean up all the messages
    (1) remove punctuation, (2) remove stopwords '''
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words(config.etl.stop_words)]
    return Test_punc_removed_join_clean
 

@task(cache_key_fn=tasks.task_input_hash, cache_expiration=datetime.timedelta(days=1))
def transform_data(df: pd.DataFrame)-> Tuple[pd.DataFrame, pd.Series]:
    # first let's drop the column
    df.drop([config.etl.drop3], axis=1, inplace=True)
    # Define the cleaning pipeline we defined earlier
    vectorizer = CountVectorizer(analyzer = message_cleaning)
    reviews_countvectorizer = vectorizer.fit_transform(df[config.etl.drop1].values.astype('U'))
    # first let's drop the column
    df.drop([config.etl.drop1], axis=1, inplace=True)
    df.drop([config.etl.drop2], axis=1, inplace=True)
    reviews = pd.DataFrame(reviews_countvectorizer.toarray())
    # Now let's concatenate them together
    df = pd.concat([df, reviews], axis=1)
    # Let's drop the target label coloumns
    X = df.drop([config.etl.target], axis=1)
    y = df[config.etl.target]
    return X, y


@task(log_prints=True, retries=3, retry_delay_seconds=5)
def load_data(df_x: pd.DataFrame, df_y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=config.etl.test_size)
    os.makedirs(config.etl.data.dst, exist_ok=True)
    X_train.to_csv(config.etl.data.train.x)
    y_train.to_csv(config.etl.data.train.y)
    X_test.to_csv(config.etl.data.test.x)
    y_test.to_csv(config.etl.data.test.y)

    local_file_1 = config.etl.data.train.x
    local_file_2 = config.etl.data.train.y
    local_file_3 = config.etl.data.test.x
    local_file_4 = config.etl.data.test.y

    uploaded_1 = upload_to_aws(local_file_1, config.etl.bucket_name, config.etl.s3_file_name_1)
    uploaded_2 = upload_to_aws(local_file_2, config.etl.bucket_name, config.etl.s3_file_name_2)
    uploaded_3 = upload_to_aws(local_file_3, config.etl.bucket_name, config.etl.s3_file_name_3)
    uploaded_4 = upload_to_aws(local_file_4, config.etl.bucket_name, config.etl.s3_file_name_4)
    print(f'file uploaded: {uploaded_1}, file2: {uploaded_2}, file3: {uploaded_3}, file4: {uploaded_4}')
    print("Data loaded to a data lake!")


# @hydra.main(version_base=None, config_path='config', config_name='main')
@flow()
def data_etl():
    df = read_data()
    df_x, df_y = transform_data(df)
    load_data(df_x, df_y)



if __name__ == '__main__':
    data_etl()
