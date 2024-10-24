import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.utils import load_object

#step 1: make the prediction class
class PredictionPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            #step 1 load the svaed model and preprocessor pickle file
            model_path=os.path.join('artifacts','model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print('Before Loading')
            #load the model and oreprocessor
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("after loading")
            data_scaled=preprocessor.transform(features)
            #make prediction
            pred=model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e,sys)
        
#step 2: create the sutom data class
class CustomData:
    def __init__(   self,
                gender:str, 
                race_ethnicity:str, 
                parental_level_of_education:str,
                lunch:str,
                test_preparation_course:str,
                math_score:int,
                reading_score:int,
                writing_score:int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.math_score=math_score
        self.reading_score=reading_score
        self.writing_score=writing_score
    #get the data as a dataframe
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender], 
                "race_ethnicity":[self.race_ethnicity], 
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "math_score":[self.math_score],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            #return the data as a dataframe
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)

