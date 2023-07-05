import shutil
import os , sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from flask import request
from src.utils import load_object
from dataclasses import dataclass

@dataclass

class PredictionFileDetial:
    prediction_output_dirname : str = 'Prediction'
    prediction_file_name:str = 'Prediction_file.csv'
    prediction_file_path : str = os.path.join(prediction_output_dirname , prediction_file_name)


class PredictionPipeline:
    def __init__(self , request:request):
        self.request = request
        self.prediction_file_detail = PredictionFileDetial()


    def save_input_file(self)->str:
        try:

            pred_file_input_dir = 'prediction_artifacts'
            os.makedirs(pred_file_input_dir , exist_ok=True)


            input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(pred_file_input_dir , input_csv_file.filename)

            input_csv_file.save(pred_file_path)

            return pred_file_path

        except Exception as e:
            logging.info("Exception occured in save_input_file")
            raise CustomException(e , sys)
        

    def predict(self , features):
        try:

            model_path = os.path.join('artifacts' , 'model.pkl')
            model = load_object(file_path=model_path)
            preprocessor = os.path.join("artifacts" , 'preprocessor.pkl')
            transform_x = preprocessor.transform(features)
            preds = model.predict(transform_x)

            return preds

        except Exception as e:
            logging.info("Excption occured  in predict")
            raise CustomException(e , sys)
        


    def get_predicted_dataframe(self , input_dataframe_path:pd.DataFrame):
        try:

            prediction_column_name : str = 'Col_8'
            input_dataframe : pd.DataFrame = pd.read_csv(input_dataframe_path)
            prediction = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in prediction]
            

            

            os.makedirs(self.prediction_file_detail.prediction_output_dirname , exist_ok=True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path , index=False)
            logging.info("Prediction Completed")


        except Exception as e:
            logging.info("Exception occured in get_predicted_dataframe")
            raise CustomException(e , sys)
        



    def run_pipeline(self):
        try:
            input_csv_path  = self.save_input_file()
            self.get_predicted_dataframe(input_csv_path) 

            return self.prediction_file_detail
        
        except Exception as e:
            logging.info("Exception occured in last run_pipeline")
            raise CustomException(e , sys)