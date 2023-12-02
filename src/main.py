from operator import index
from datetime import datetime
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import mlflow
import pandas as pd
from src.models.weatherModel import TransactionModel
from src.preprocessing.data_processing_json import clean_data_json
from frontend.app import main as st_main


# Set MLflow tracking credentials and experiment information
os.environ['MLFLOW_TRACKING_USERNAME'] = "Sahar-dev"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "1a82d2e1cf9c21f919dfb44e98e6eb57fc75ab0a"
mlflow.set_tracking_uri("https://dagshub.com/Sahar-dev/weather.mlflow")
mlflow.set_experiment("Final-experiment")

# Load the MLflow model
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments, filter_string="metrics.F1_score_test <1")
run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']
logged_model = f'runs:/{run_id}/ML_models'
model = mlflow.pyfunc.load_model(logged_model)

# Create FastAPI app instance
app = FastAPI()

# Define CORS middleware to allow cross-origin requests
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files (if any)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the root endpoint to serve the HTML page
@app.get("/")
async def read_root():
    with open("frontend/index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)
@app.get("/streamlit", response_class=HTMLResponse)
async def run_streamlit(request: Request):
    # Run the Streamlit app logic from app.py
    st_main()
    return HTMLResponse(content="", status_code=200)
# Define the predict endpoint
@app.post("/predict")
def predict(data: TransactionModel):
    """
    Endpoint to make predictions using the MLflow model.
    
    Parameters:
    - data: TransactionModel - Input data for prediction.

    Returns:
    - dict: Predicted values.
    """
    # Convert input data to DataFrame
    received = data.dict()
    df = pd.DataFrame(received, index=[0])

    # Preprocess the data
    preprocessed_data = clean_data_json(df)

    # Make predictions using the loaded MLflow model
    predictions = model.predict(preprocessed_data)

    return {"predictions": predictions.tolist()}
