import os
import sys
from dataclasses import dataclass
# from catboost import CatBoostclassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearClassification
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('split training and test input data')
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                # "XGBClassifer": XGBClassifier(),
                # "CatBooSt": CatBoostclassifier(iterations=100),
                "Linear Classifier":LinearClassification()
            }
            model_repot:dict = evaluate_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.vlaues()).index(best_model_score)


            ]
            best_model= models[best_model_name]
            if(best_model_score < 0.6):
                raise CustomException("No best model found")
            logging.info("Best found model on both training and testing data")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model_predict(x_test)
            r2_square = r2_score(y_tes, prediction)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
