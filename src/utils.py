import os
import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)
    




def evaluate_models(trainx, trainy,testx,testy,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(trainx,trainy)

            model.set_params(**gs.best_params_)
            model.fit(trainx,trainy)



            y_train_pred = model.predict(trainx)

            y_test_pred = model.predict(testx)

            train_model_score = r2_score(trainy, y_train_pred)

            test_model_score = r2_score(testy, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)