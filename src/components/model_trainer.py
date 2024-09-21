## we will use top 5 best performace ML algorithm
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
from dataclasses import dataclass
from sklearn.metrics import r2_score
import os,sys

@dataclass 
class ModelTrainerConfig:
    trained_model_file = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_mode_trainer(self,train_array,test_array):
        try:
            logging.info("split train and test input data")
            x_train, y_train, x_test, y_test = ( train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1] )
            models = {
                "LinearRegression":LinearRegression(),
                "Ridge":Ridge(),
                "RandomForestRegressor":RandomForestRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=0)
            }
            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model found")

            logging.info("Best Model found on both train and test")

            save_object(
                file_path=self.model_trainer_config.trained_model_file,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2 = r2_score(y_test,predicted)

            return r2

        except Exception as e:
            raise CustomException(e,sys)