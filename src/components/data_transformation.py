import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class data_transform_config:
    def __init__(self):
        self.data_transformation_info=os.path.join('artifacts','preprocessor.pkl')


class data_transformation:
    def __init__(self):
        self.data_transform=data_transform_config()

    def get_data_transform(self):

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]


            num_pipline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler(with_mean=False))])
            cat_pipline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),('encoding', OneHotEncoder()),('scale',StandardScaler(with_mean=False))])
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")


            preprocessor=ColumnTransformer([('num_pipline',num_pipline,numerical_columns),
                                            ('cat_pipline',cat_pipline,categorical_columns)])
            
            logging.info('data transformation preprocessing pipline sussessfully completed')

            return preprocessor


        except Exception as e:
            CustomException(e,sys)


    def initiate_data_transform(self,train_path,test_path):
        
        try:
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)

            target_col=['math_score']
            numericaLl_col=["writing_score", "reading_score"]

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transform()


            input_train_data=train.drop(target_col,axis=1)
            output_train_data=train[target_col]

            input_test_data=test.drop(target_col,axis=1)
            output_test_data=test[target_col]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe.')


            train_data=preprocessing_obj.fit_transform(input_train_data)
            test_data=preprocessing_obj.fit_transform(input_test_data)

            logging.info('fit and transform completed.')


            train_arr = np.c_[train_data,np.array(output_train_data)]
            test_arr=np.c_[test_data,np.array(output_test_data)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transform.data_transformation_info,
                obj=preprocessing_obj)

            return (train_arr,test_arr,self.data_transform.data_transformation_info)



        except Exception as e:
            CustomException (e,sys)
    


