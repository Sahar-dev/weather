from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import sklearn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
from weatherModel import TransactionModel
from data_processing_json import clean_data_json
import os
from fastapi.responses import HTMLResponse
os.environ['MLFLOW_TRACKING_USERNAME']= "Sahar-dev"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "1a82d2e1cf9c21f919dfb44e98e6eb57fc75ab0a"
mlflow.set_tracking_uri("https://dagshub.com/Sahar-dev/weather.mlflow")
mlflow.set_experiment("Final-experiment")
# Fetch all experiments
# all_experiments = mlflow.search_experiments()

# all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
# df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.f1_score <1")
# run_id = df_mlflow.loc[df_mlflow['metrics.f1_score'].idxmax()]['run_id']
# print (all_experiments)
# print (run_id)
app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.F1_score_test <1")
run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']

logged_model = f'runs:/{run_id}/ML_models'

model = mlflow.pyfunc.load_model(logged_model)


@app.get("/")
async def read_root():
    with open("C:/Users/sahas/weather/frontend/dashboard.html", "r") as file:
         return HTMLResponse(content=file.read(), status_code=200)

# this endpoint receives data in the form of json (informations about one transaction)
@app.post("/predict")
def predict(data : TransactionModel):
    received = data.dict()
    df =  pd.DataFrame(received,index=[0])
    preprocessed_data = clean_data_json(df)
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}





# artifact_uri = f'runs:/{run_id}/ML_models'
# print("Artifact URI:", artifact_uri)

# logged_model = f'runs:/{run_id}/ML_models'
# model = mlflow.pyfunc.load_model(logged_model)



# all_experiments = [7]
# #print(all_experiments)
# df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.f1_score <1")
# run_id = df_mlflow.loc[df_mlflow['metrics.f1_score'].idxmax()]['run_id']
# #print(run_id)
# logged_model = f'runs:/{run_id}/ML_models_testing_3'
# # print(logged_model)
# model = mlflow.pyfunc.load_model(logged_model)


# @app.get("/")
# def read_root():
#     return {"Hello": "to Rain detector app"}

# def get_index_content():
#     with open("C:/Users/sahas/weather/src/dashboard.html", "r") as file:
#         return file.read()

# @app.get("/", response_class=HTMLResponse)
# async def read_root():
#     html_content = get_index_content()
#     return HTMLResponse(content=html_content)
    
