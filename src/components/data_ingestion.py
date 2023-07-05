from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
import os , sys

@dataclass

class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts' , 'train.csv')
    test_data_path:str = os.path.join("artifacts" , 'test.csv')
    raw_data_path:str = os.path.join("artifacts" , "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data engestion method started")

        try:
            df = pd.read_csv(os.path.join('notebooks' , 'cement2.csv'))

            logging.info("Data is collected")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path) , exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path , index=False , header=True)

            logging.info("Database read as DataFrame form")

            train_set , test_set = train_test_split(df  , test_size=0.32 , random_state=564)

            logging.info("train test split is completed")

            train_set.to_csv(self.ingestion_config.train_data_path , index=False , header=True)
            test_set.to_csv(self.ingestion_config.test_data_path , index=False , header=True)

            logging.info("Ingestion of data in completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured in initiate_data_ingestion")
            raise CustomException(e , sys)