import sys,os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
            this is the function for getting transformation object
        '''
        try:
            categorical_features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            numerical_features = ['reading score', 'writing score']
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))  
                ]
            )

            logging.info("numerical columns standard Scalling completed")
            logging.info("categorical columns encoding completed")

            preprocessor = ColumnTransformer([
                ("num_pipeline",numerical_pipeline,numerical_features),
                ("cat_pipeline",categorical_pipeline,categorical_features)
            ])

            return preprocessor
        

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info("reading train and test data completed")
            logging.info("obtaining preproccessor object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "math score"
            numerical_column= []

            input_feature_train_df = train_df.drop([target_column_name],axis=1)
            target_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop([target_column_name],axis=1)
            target_test_df = test_df[target_column_name]
            logging.info("appyling preprocessing on the train and test data")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            train_arr = np.c_[input_feature_train_arr,np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_test_df)]
            logging.info("saved Preproccessing Object")
            save_object( 
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            return (
                train_arr,test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e,sys)
