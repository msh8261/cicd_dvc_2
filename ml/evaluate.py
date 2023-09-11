import warnings
warnings.filterwarnings(action="ignore")
import os
import sys
sys.path.insert(0, '..')
import numpy as np
from dvclive import Live
import hydra
from hydra.utils import to_absolute_path as abspath
from hydra import initialize, compose
from prefect import task, flow
from omegaconf import DictConfig
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple

from store.read_write_s3  import read_s3_file, upload_to_aws

 
with initialize(version_base=None, config_path="../config"):
    config = compose(config_name="main", overrides=["ml=ml1"])    
    # print(OmegaConf.to_yaml(config))



@task
def load_data()-> Tuple: 
    # Load data
    df_x = read_s3_file(config.ml.bucket_name, 'xtest')
    X_test = pd.read_csv(df_x)
    df_y = read_s3_file(config.ml.bucket_name, 'ytest')
    y_test = pd.read_csv(df_y)
    # X_test = pd.read_pickle(abspath(config.etl.data.test.x))
    # y_test = pd.read_pickle(abspath(config.etl.data.test.y))
    return X_test, y_test


@task
def load_model():
    # Load model
    model_File = read_s3_file(config.ml.bucket_name, 'nbm_sentiment_analysis')
    model = joblib.load(model_File)
    # model_name = config.ml.model.name
    # dir_m = config.ml.model.path
    # local_file = os.path.join(dir_m, model_name)
    # model = joblib.load(abspath(local_file))
    return model


# @hydra.main(version_base=None, config_path="../config", config_name="main")
@flow
def main() -> None:
    """Evaluate model and log metrics"""       
    with Live(save_dvc_exp=True, resume=True) as live:
        X_test, y_test= load_data()
        model = load_model()
        y_pred = model.predict(X_test)
        # y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test,axis=1)
        # Get metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy Score of this model is {accuracy}.")
        os.makedirs('dvclive', exist_ok=True)
        live.log_metric("accuracy", accuracy)



if __name__ == "__main__":
    main()
