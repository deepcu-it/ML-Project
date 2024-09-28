import os,sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(x_train,y_train,x_test,y_test,models:dict)->dict:
    try:
        report_model: dict = {}
        for modelname in models.keys():
            model = models[modelname]

            model.fit(x_train,y_train)

            y_test_pred = model.predict(x_test)

            r2_test_score = r2_score(y_test,y_test_pred)

            report_model[modelname] = r2_test_score
        
        return report_model
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_hyperparametertunning(x_train,y_train,x_test,y_test,models:dict,params:dict)->dict:
    try:
        report_model: dict = {}
        for modelname in models.keys():
            print(modelname)
            obj = GridSearchCV(estimator=models[modelname],param_grid=params[modelname],verbose=0,error_score='raise')
            obj.fit(x_train,y_train)
            y_pred = obj.predict(x_test)

            r2_test_score = r2_score(y_test,y_pred)

            report_model[modelname] = r2_test_score
        
        return report_model
    except Exception as e:
        raise CustomException(e,sys)
    
