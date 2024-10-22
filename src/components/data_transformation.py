import pandas as pd
import numpy as np
import os


import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    #save the tranfromer object so it can be loaded later during model training and prediction
    preprocessor_obj_file_path:str=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.DataTransformation_config=DataTransformationConfig()

    def get_data_tranformer_object(self):
        try:
            #step 1: Define numerical and categorical variables 
            numerical_cols= ['math score', 'reading score', 'writing score']
            categorical_cols= ['gender', 
                               'race/ethnicity', 
                               'parental level of education',
                                 'lunch',
                                 'test preparation course'
                                 ]
           
            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("Onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"categorical colums: {categorical_cols}")
            logging.info(f"numerical colums: {numerical_cols}")

            preprocessor=ColumnTransformer(
                [
                ("numerical_pipeline",num_pipeline,numerical_cols),
                ("categorical_pipeline",cat_pipeline,categorical_cols)
                ]

            )
            logging.info("Data transformer object created successfully")
            return preprocessor

        except Exception as e:
            logging.info("Error in creating data transformer object:")
            raise CustomException (e, sys)
    def initiate_data_transformation(self,train_path, test_path):
        try:
            #step1: read train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read and train data completed")
            logging.info(f"Train DataFrame shape: {train_df.shape}, Test DataFrame shape: {test_df.shape}")
            #step 2: extract features and target variables
            target_column ='total_score'
            input_feature_train=train_df.drop(columns=target_column,axis=1)
            target_train=train_df[target_column]

            input_feature_test=test_df.drop(columns=target_column,axis=1)
            target_test=test_df[target_column]

            #step 3 get the preproccesor object
            
            logging.info("Obtaining preprocceing data")
            preprocessor_obj = self.get_data_tranformer_object()
            
            input_feature_train_transformed=preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_transformed=preprocessor_obj.transform(input_feature_test)
            logging.info("Data transformation completed successfully.")
            train_arr = np.c_[
                input_feature_train_transformed, np.array(target_train)
            ]
            test_arr = np.c_[input_feature_test_transformed, np.array(target_test)]

            #step 4: svae the preprocessor obj for future use
            save_object(file_path=self.DataTransformation_config.preprocessor_obj_file_path,
                                  obj=preprocessor_obj)
            return (train_arr, 
                    test_arr, 
                    self.DataTransformation_config.preprocessor_obj_file_path,)

            
        except Exception as e:
            logging.error("Error in the data transformation process.")
            raise CustomException(e, sys)

            

