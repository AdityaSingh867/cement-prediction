import os , sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from dataclasses import dataclass


@dataclass

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts" , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_the_transformation_object(self):
        try:
            logging.info("Data transformation initiated")
            cols_for_scaling = [
                'Col_0', 'Col_1', 'Col_2', 'Col_3', 'Col_4', 'Col_5', 'Col_6', 'Col_7'
            ]

            scaler_pipeline = Pipeline(steps=[
                ('scaler' , StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ('scaler' , scaler_pipeline , cols_for_scaling)
            ])

            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in get_the_transformation_object")
            raise CustomException(e , sys)
        
    def initiate_data_transformation(self , train_path , test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data is completed")

            preprocessing_obj = self.get_the_transformation_object()

            target_column = ['Col_8']

            ### Training DataFrame
            
            input_feature_train_df = train_df.drop(target_column , axis=1)
            target_feature_train_df = train_df[target_column]

            ### Testing DataFrame
            input_feature_test_df = test_df.drop(target_column , axis=1)
            target_feature_test_df = test_df[target_column]

            ## Transform train and test data
            transform_input_train_feature = preprocessing_obj.fit_transform(input_feature_train_df)
            transform_input_test_feature = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[transform_input_train_feature , np.array(target_feature_train_df)]
            test_arr = np.c_[transform_input_test_feature , np.array(target_feature_test_df)]

            save_obj(
                self.data_transformation_config.preprocessor_obj_file_path , 
                obj=preprocessing_obj
            )

            return(
                train_arr , test_arr
            )



        except Exception as e:
            logging.info("Exception occured in initiate_data_transformation")
            raise CustomException(e , sys)