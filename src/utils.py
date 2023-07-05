import pandas as pd
import numpy as np
import pickle , os , sys
from sklearn.metrics import r2_score , mean_squared_error
from src.logger import logging
from src.exception import CustomException

def save_obj(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path , exist_ok=True)
        with open(file_path , 'wb') as file_obj:
            pickle.dump(obj , file_obj)
    except Exception as e:
        logging.info("Exception occured in save_obj")
        raise CustomException(e , sys)
    

def evaluate_model(X_train , y_train , X_test , y_test , models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            ## Train Model
            model.fit(X_train , y_train)
            ## Prediction 
            y_pred = model.predict(X_test)
            ## Calculate score
            test_model_score = r2_score(y_test , y_pred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        logging.info("Exception occured in evaluate_model")
        raise CustomException(e , sys)
    
    
def load_object(file_path):
    try:
        with open(file_path , 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Exception occured in load_object")
        raise CustomException(e , sys) 