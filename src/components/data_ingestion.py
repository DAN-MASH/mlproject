import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    #artifacts folder stores all the output
    # data ingestion component now knows where to save the train, test and raw data file path
    raw_data_path: str= os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str= os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

#start class data ingestion
class DataIngestion:
    def __init__(self):
        #the three patths get saved into this class variable
        self.ingestion_config=DataIngestionConfig()

        #reads the data
    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion")
        try:
            df=pd.read_csv('notebook\data\stud1.csv')
            logging.info("Read the data as a dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            #save the data
            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)

            #splitting the data into train,and test
            logging.info("Initiate train_test_split")
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=42)
            logging.info(f"Train_set shape {train_set.shape}")
            logging.info(f"Train_set shape {test_set.shape}")
            #save train and test data as csv
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info(f"Error occured during data ingestion: {e}")
            raise CustomException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    logging.info(f"Data ingestion completed. Train data at: {train_data}, Test data at: {test_data}")

    data_transformation=DataTransformation()
    train_array,test_array,_=data_transformation.initiate_data_transformation(train_data,test_data)

