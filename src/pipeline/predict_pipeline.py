import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Age:float,
        Gender:str,
        Location:str,
        Subscription_Length_Months:float,
        Monthly_Bill:float,
        Total_Usage_GB:float):

        self.Age=Age
        self.Gender=Gender
        self.Location=Location
        self.Subscription_Length_Months=Subscription_Length_Months
        self.Monthly_Bill=Monthly_Bill
        self.Total_Usage_GB=Total_Usage_GB

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Gender':[self.Gender],
                'Location':[self.Location],
                'Subscription_Length_Months':[self.Subscription_Length_Months],
                'Monthly_Bill':[self.Monthly_Bill],
                'Total_Usage_GB':[self.Total_Usage_GB]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)


#import sys; print("Python", sys.version)
#import numpy; print("NumPy", numpy.__version__)
#import scipy; print("SciPy", scipy.__version__)
#import sklearn; print("Scikit-Learn", sklearn.__version__)