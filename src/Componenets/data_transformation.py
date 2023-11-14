import sys
from dataclasses import dataclass

sys.path.append(r'D:\FSDS_2.0\python\Machine_learning\Projects\diamond_price_prediction\src')

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from exception import custom_exception
from logger import logging
from utils import save_obj
import os


@dataclass
class data_transforamtion_config:

    preprocessor_obj_file_path = os.path.join(r'../../Artifacts', 'preprocessor.pkl')


class data_transformation:


    def __init__(self):
        self.data_transformation_config = data_transforamtion_config()


    def get_data_transformation_obj(self):
        try:
            logging.info('Creating Data Transformation Object')

            # Reading Raw Dataset
            raw_df = pd.read_csv('../../Artifacts/raw.csv')

            # Seperating Categorical and Numerical Features
            cat_cols = [col_name for col_name in raw_df.columns if raw_df[col_name].dtype == 'object']
            num_cols = [col_name for col_name in raw_df.columns if raw_df[col_name].dtype != 'object']

            num_cols = num_cols[1:len(num_cols)-1]

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Data Transformation PipeLine Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline 
            # Need to do scaling since we are using ordinal encoding 
            # If we had used one hot encoding there was no need of scaling

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinal_encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scalar', StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
            ])
        
            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info('Error Occured In Data Transfromation')
            raise custom_exception(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info('Data Transformation Initiated')

            logging.info('Reading train and test data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('train data head\n {}'.format(train_df.head().to_string()))
            logging.info('test data head\n {}'.format(test_df.head().to_string()))
            
            logging.info('Obtaining Preprocessor Object')
            preprocessor_obj = self.get_data_transformation_obj()


            #seperating input and output features for train and test datasets

            target_column_name = 'price'
            drop_columns = [target_column_name,'id']

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                object=preprocessor_obj
                )

            logging.info('Preprocessor Pickel File saved')

            return(

                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path       

            )

        except Exception as e:

            logging.info('Exception has occured in data transformation')
            raise custom_exception(e, sys)


# if __name__ == '__main__':
#     data_trans_obj = data_transformation()
#     data_trans_obj.get_data_transformation_obj()