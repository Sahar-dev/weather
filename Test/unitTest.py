from pathlib import Path
import pickle
import pandas as pd
import sys
from src.preprocessing.data_processing_json import clean_data_json
from src.preprocessing.data_processing import transform_data
import mlflow
#from dotenv import load_dotenv
import os

#from dotenv import load_dotenv

#load_dotenv(".env")

#DagsHub_username = os.getenv("DagsHub_username")
#DagsHub_token=os.getenv("DagsHub_token")

#DagsHub_username = os.environ["DAGSHUB_USERNAME"]
#DagsHub_token=os.environ["DAGSHUB_TOKEN"]

os.environ['MLFLOW_TRACKING_USERNAME']= "Sahar-dev"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "1a82d2e1cf9c21f919dfb44e98e6eb57fc75ab0a"
mlflow.set_tracking_uri("https://dagshub.com/Sahar-dev/weather.mlflow")

mlflow.set_experiment("Final-experiment")
#tests if the model works as expected

def test_model_use():
    #let's call the model from the model registry ( in production stage)

    #let's call the model from the model registry ( in production stage)

    df_mlflow=mlflow.search_runs(filter_string="metrics.F1_score_test < 1")
    print(df_mlflow.columns)
    run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']



    logged_model = f'runs:/{run_id}/ML_models'

    # Load model as a PyFuncModel.
    model = mlflow.pyfunc.load_model(logged_model)

    d = {
    "Date": "2008-12-01",
    "Location": "Albury",
    "MinTemp": 13.4,
    "MaxTemp": 22.9,
    "Rainfall": 0.6,
    "Evaporation": 12,
    "Sunshine": 12,
    "WindGustDir": "W",
    "WindGustSpeed": 44,
    "WindDir9am": "W",
    "WindDir3pm": "WNW",
    "WindSpeed9am": 20,
    "WindSpeed3pm": 24,
    "Humidity9am": 71,
    "Humidity3pm": 22,
    "Pressure9am": 1007.7,
    "Pressure3pm": 1007.1,
    "Cloud9am": 8,
    "Cloud3pm": 10,
    "Temp9am": 16.9,
    "Temp3pm": 21.8,
    "RainToday": "No"
    }

    df = pd.DataFrame(data=d,index=[0])
    dd = clean_data_json(df)
    predict_result = model.predict(dd)
    print(predict_result[0])
    assert predict_result[0] == 1

test_model_use()
    