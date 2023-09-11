import sys
sys.path.insert(0, '..')
import os
import joblib
import pandas as pd
import logging
from typing import Dict, Text, Any
from pathlib import Path

from evidently import ColumnMapping
from evidently.metrics import RegressionErrorDistribution
from evidently.metrics import RegressionErrorPlot
from evidently.metrics import RegressionQualityMetric
from evidently.report import Report

import hydra
from hydra import compose, initialize 
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

# from s3.s3_bucket import read_s3_file, upload_to_aws

import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


logging.basicConfig(
    level=logging.INFO,
    format='Monitoring - %(asctime)s - %(levelname)s - %(message)s')


with initialize(version_base=None, config_path="../config"):
    config = compose(config_name="main")
 

@task
def load_data() -> pd.DataFrame:
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


@task
def load_model():
    # Load model
    # model_file = read_s3_file(config.ml.bucket_name, 'nbm_sentiment_analysis')
    # model = joblib.load(model_file)
    model_name = config.ml.model.name
    dir_m = config.ml.model.path
    local_file = os.path.join(dir_m, model_name)
    model = joblib.load(local_file)
    return model


def build_model_monitoring_report(
                                    reference_data: pd.DataFrame,
                                    current_data: pd.DataFrame,
                                    column_mapping: ColumnMapping,
                                ) -> Any:

    model_report = Report(
                            metrics=[
                                RegressionQualityMetric(),
                                RegressionErrorPlot(),
                                RegressionErrorDistribution(),
                            ]
                        )
    model_report.run(
                        reference_data=reference_data,
                        current_data=current_data,
                        column_mapping=column_mapping,
                    )

    return model_report



def get_model_monitoring_metrics(regression_quality_report: Report) -> Dict:
    metrics = {}
    report_dict = regression_quality_report.as_dict()
    metrics["me"] = report_dict["metrics"][0]["result"]["current"]["mean_error"]
    metrics["mae"] = report_dict["metrics"][0]["result"]["current"]["mean_abs_error"]
    metrics["rmse"] = report_dict["metrics"][0]["result"]["current"]["rmse"]
    metrics["mape"] = report_dict["metrics"][0]["result"]["current"]["mean_abs_perc_error"]
    return metrics


def get_column_mapping():
    column_mapping = ColumnMapping()
    dc = config.vis.DATA_COLUMNS
    column_mapping.FEATURE_COLUMNS = dc['num_features'] + dc['cat_features']
    column_mapping.target = dc['target']
    column_mapping.prediction = dc['prediction']
    column_mapping.numerical_features = dc['num_features']
    column_mapping.categorical_features = dc['cat_features']
    return column_mapping


def define_reference_data(model, df: pd.DataFrame):
    logging.info('Read reference data')
    reference_data = df[500:].sample(100, random_state=0)
    return reference_data


def define_current_data(df: pd.DataFrame):
    logging.info('Read current data') 
    current_data = df[:500].sample(100, random_state=0)
    return current_data


def get_reports(reference_data, current_data):
    logging.info('Build model performance report')
    column_mapping = get_column_mapping()
    model_report: Text = build_model_monitoring_report(
                                                reference_data=reference_data,
                                                current_data=current_data,
                                                column_mapping=column_mapping
                                            )
    model_metrics = get_model_monitoring_metrics(model_report)
    return model_report, model_metrics 


@task
def model_quality_evaluation(model, df):
   
    reference_data = define_reference_data(model, df)        
    
    # Make predictions for the current batch data
    current_data = define_current_data(df)

    # reference_prediction = model.predict(reference_data[DATA_COLUMNS['num_features'] + DATA_COLUMNS['cat_features']])
    # reference_data['prediction'] = reference_prediction

    #current_prediction = model.predict(current_data[DATA_COLUMNS['num_features']])
    #current_data['prediction'] = current_prediction


    # Build the Model Monitoring report
    model_report, model_metrics = get_reports(reference_data, current_data)
        
    # Log Monitoring Report 
    os.makedirs(config.vis.REPORTS_DIR, exist_ok=True)
    monitoring_report_path = f"{config.vis.REPORTS_DIR}/{config.vis.REPORT_NAME}"
    model_report.save_html(monitoring_report_path)
 


@flow(task_runner=SequentialTaskRunner())
def main():
    #load data
    df = load_data()
    # load model
    model = load_model()  
        
    model_quality_evaluation(model, df)


if __name__ == "__main__":
    main()



















