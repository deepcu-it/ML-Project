import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parent_education,lunch:str,test_prep_score:str,reading_score:int,writing_score:int) -> None:
        self.gender = gender
        self.race_ethnicity=race_ethnicity
        self.parent_education = parent_education
        self.lunch = lunch
        self.test_prep_score= test_prep_score
        self.reading_score = reading_score
        self.writing_score = writing_score
    def get_data_as_dataframe(self):
        data = {
            'gender': [self.gender],
            'race/ethnicity': [self.race_ethnicity],
            'parental level of education': [self.parent_education],
            'lunch': [self.lunch],
            'test preparation course': [self.test_prep_score],
            'reading score': [self.reading_score],
            'writing score': [self.writing_score]
        }
        df = pd.DataFrame(data)
        return df

class PredictPipeline:
    def __init__(self) -> None:
        pass
    def predict(self,features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            raise CustomException(e,sys)