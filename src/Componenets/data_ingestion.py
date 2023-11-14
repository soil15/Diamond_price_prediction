import os
import sys
sys.path.append(r'D:\FSDS_2.0\python\Machine_learning\Projects\diamond_price_prediction\src')
from exception import custom_exception
from logger import logging
from Componenets.data_transformation import data_transformation
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


# Data Ingestion Configuration
@dataclass
class data_ingestion_config:

    train_data_path:str = os.path.join('../../Artifacts', 'train.csv')
    test_data_path:str = os.path.join('../../Artifacts', 'test.csv')
    raw_data_path:str = os.path.join('../../Artifacts', 'raw.csv')


# Create Class for Data Ingestion

class Data_ingestion:

    def __init__(self):
        self.ingestion_config = data_ingestion_config()

    
    def initiate_data_ingestion(self):

        logging.info('Data Ingestion method has initiated')

        try:
            
            # sys.path.append(r'D:\FSDS_2.0\python\Machine_learning\Projects\diamond_price_prediction\notebooks\data')
            df = pd.read_csv(r'../../notebooks/data/gemstone.csv')
            logging.info('Dataset Read as Pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Initiating Train test Split')
            train_set, test_set = train_test_split(df, test_size=0.30, random_state=40)

            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info('Data Ingestion Has Completed')

            return (

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        
        except Exception as e:

            logging.info('Exception Has Occured During Data Ingestion')
            raise custom_exception(e, sys)



if __name__ == '__main__':

    data_ing_obj = Data_ingestion()
    train_data_path, test_data_path = data_ing_obj.initiate_data_ingestion()

    data_transformation = data_transformation()
    train_arr, test_arr, pkl_file_path  = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
