import os
import sys
sys.path.insert(0, '..')
import joblib
import pickle
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import hydra
from hydra import initialize, compose
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from s3.s3_bucket import read_s3_file, upload_to_aws


with initialize(version_base=None, config_path="../config"):
    config = compose(config_name="main", overrides=["etl=etl1", "ml=ml1"])    
    # print(OmegaConf.to_yaml(config))


def create_pipeline() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svc", SVC())])


def train_model(
                X_train: pd.DataFrame,
                y_train: pd.Series,
                pipeline: Pipeline,
                hyperparameters: dict,
                grid_params: dict,
            ) -> GridSearchCV:
    """Train model using GridSearchCV"""
    grid_search = GridSearchCV(pipeline, dict(hyperparameters), **grid_params)
    grid_search.fit(X_train, y_train)
    return grid_search


@task
def train(config, X_train, y_train) -> None:
    """Train model and save it"""  
    pipeline = create_pipeline()
    grid_search = train_model(
                                X_train,
                                y_train,
                                pipeline,
                                config.ml.model.hyperparameters,
                                config.ml.model.grid_search,
                            )

    return grid_search



# @hydra.main(version_base=None, config_path="../config", config_name="main")
@flow(task_runner=SequentialTaskRunner())
def main():
    df_x = read_s3_file(config.ml.bucket_name, 'xtrain')
    X_train = pd.read_csv(df_x)
    df_y = read_s3_file(config.ml.bucket_name, 'ytrain')
    y_train = pd.read_csv(df_y)
    # X_train = pd.read_csv(config.etl.data.train.x)
    # y_train = pd.read_csv(config.etl.data.train.y)
    grid_search = train(config, X_train, y_train['feedback'])
    # save model
    model_name = config.ml.model.name
    dir_m = config.ml.model.path
    os.makedirs(dir_m, exist_ok=True)
    local_file = os.path.join(dir_m, model_name)
    with open(local_file, 'wb') as fout:
        pickle.dump(grid_search, fout)   
    uploaded_model = upload_to_aws(local_file, config.ml.bucket_name, config.ml.s3_model_name)
    print(f'file uploaded: {uploaded_model}')



if __name__ == "__main__":
    main()
