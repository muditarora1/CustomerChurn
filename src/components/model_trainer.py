import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "XGBClassifier": XGBClassifier(), 
                "Logistic":LogisticRegression(),
                "AdaBoost Classifier": AdaBoostClassifier()
            }   

            # I will be focusing on Random forest thats why I commented other model params to save time
            params={
                "Decision Tree": {
                    #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Classifier":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['auto','sqrt','log2',None],
                    'n_estimators' : [5,10,100,150],
                    'criterion': ['gini','entropy'],
                    'max_depth': range(1,5),
                    'min_samples_split' : [5,10,15,100,150],
                    'min_samples_leaf': range(1,5)
                },
                "Logistic":{},
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    #'learning_rate':[.1,.01,.05,.001],
                    #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    #'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Classifier":{
                    #'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    #'n_estimators': [8,16,32,64,128,256]
                },
                "XGBClassifier":{
                    #'learning_rate':[.1,.01,.05,.001],
                    #'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.4:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name
            

            
        except Exception as e:
            raise CustomException(e,sys)