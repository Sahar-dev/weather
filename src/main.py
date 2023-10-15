from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import pandas as pd
from data_processing import transform_data  # Update with your actual preprocessing function
# Set the MLflow tracking URI
import os

os.environ['MLFLOW_TRACKING_USERNAME']= "Sahar-dev"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "1a82d2e1cf9c21f919dfb44e98e6eb57fc75ab0a"
mlflow.set_tracking_uri("https://dagshub.com/Sahar-dev/weather.mlflow")

# Fetch all experiments
all_experiments = mlflow.search_experiments()

all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.f1_score <1")
run_id = df_mlflow.loc[df_mlflow['metrics.f1_score'].idxmax()]['run_id']
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
artifact_uri = f'runs:/{run_id}/ML_models'
print("Artifact URI:", artifact_uri)

# logged_model = f'runs:/{run_id}/ML_models'
# # model = mlflow.pyfunc.load_model(logged_model)

# @app.get("/")
# def read_root():
#     return {"Hello": "to fraud detector app"}