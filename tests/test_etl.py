import os 
import sys
sys.path.insert(0, '..')
import pandas as pd
from pathlib import Path
from hydra import initialize, compose
from typing import Tuple

from etl.etl import *


with initialize(version_base=None, config_path="../config"):
    config = compose(config_name="main", overrides=["etl=etl1", "ml=ml1", "vis=vis1"])    

# def test_raw_data_exist():
#     assert os.path.exists(config.etl.data.raw)


# def test_read_data():
#     # for prefect function use fn()
#     df = read_data.fn()
#     assert type(df) == pd.DataFrame


def test_message_cleaning():
    mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'
    expected_string  = ['mini', 'challenge', 'teach', 'remove', 'stopwords', 'punctuations']
    cleaned_string = message_cleaning(mini_challenge)
    assert expected_string == cleaned_string

def test_transform_data():
    # create dataset
    df = pd.DataFrame({'variation': ['black dot', 'black dot', 'whit dot'],
                        'date' : ['01-10-2020', '02-10-2020', '03-10-2020'],
                    'verified_reviews': ['love it', 'bad as bad', 'good one'],
                    'feedback': [1, 0, 1]})
    df_x, df_y = transform_data.fn(df)
    assert type(df_x) == pd.DataFrame
    assert type(df_y) == pd.Series









