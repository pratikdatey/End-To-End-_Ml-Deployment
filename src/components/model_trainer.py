import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models




@dataclass
class model_trainer_path_config:
        train_model_file_path= os.path.join('artifacts', 'model_trainer.pkl')

class model_training:
    def __init__(self):
        self.model_config=model_trainer_path_config()

    def initialize_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Initializing train_test split')
            trainx,trainy,testx,testy=(train_arr[:,:-1],
                                       train_arr[:,-1],
                                       test_arr[:,:-1],
                                       test_arr[:,-1])
            

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()}
            

            model_report:dict=evaluate_models(trainx=trainx,trainy=trainy,testx=testx,testy=testy,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_config.train_model_file_path,
                obj=best_model)
            

            predicted=best_model.predict(testx)

            r2_square = r2_score(testy, predicted)


            return r2_square


        except Exception as e:
            CustomException(e,sys)
