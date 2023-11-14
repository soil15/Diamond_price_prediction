import sys
import os
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from exception import custom_exception
from logger import logging


def save_obj(file_path, object):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(object, file_obj)

    except Exception as e:

        logging.info('Exception occured at utils-> function ->save_object')
        raise custom_exception(e, sys)