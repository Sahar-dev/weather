
from fastapi import FastAPI


import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from fastapi import FastAPI, File, UploadFile
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import os


os.environ['MLFLOW_TRACKING_USERNAME']= "Sahar-dev"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "1a82d2e1cf9c21f919dfb44e98e6eb57fc75ab0a"


#setup mlflow
#mlflow.set_tracking_uri('https://dagshub.com/Sahar-dev/mlops_proj') #your mlfow tracking uri
#mlflow.set_experiment("idsd-sd-experiment")
app =FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "to fraud detector app"}







#uvicorn src.main:app --reload --port 8500