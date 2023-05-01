import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import data_transformation,data_transform_config


@dataclass
class Data_ingesction_config:
    train_data_path:str=os.path.join('artifacts','train_data.csv')
    test_data_path:str=os.path.join('artifacts','test_data.csv')
    raw_data_path:str=os.path.join('artifacts','raw_data.csv')


class Ingestion:
    def __init__(self):
        self.data_config=Data_ingesction_config()

    def initiate_data_ingestion(self):
        logging.info('initiating data ingestion process')
        try:
            data=pd.read_csv('notebook/data/stud.csv')
            os.makedirs(os.path.dirname(self.data_config.train_data_path),exist_ok=True)
            data.to_csv(self.data_config.raw_data_path,index=False,header=True)
            logging.info('Raw data import successfully')

            logging.info('Train Test split initiated successfully')
            train_data,test_data=train_test_split(data,test_size=0.2,random_state=42)
            train_data.to_csv(self.data_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.data_config.test_data_path,index=False,header=True)

            logging.info('Train Test split done successfully')

            return(self.data_config.train_data_path,
               self.data_config.test_data_path) 
        
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__=='__main__':
    obj=Ingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transform=data_transformation()
    data_transform.initiate_data_transform(train_data,test_data)






