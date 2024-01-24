# Rainfall Prediction MLops Project

Predict next-day rainfall in Australia using machine learning and MLOps using [Kaggle dataset]( https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)

## Features

- Data Cleaning
- Feature Selection
- Model Training
- Keeping track of the models using MLFlow
- API with FastAPI
- Frontend with Streamlit
- Dashboard with Streamlit, HTML, and PowerBI
- Testing with DeepCheck

## Setup

1. **Backend API:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
2. **Frontend**
   ```bash
   streamlit run frontend/main.py
## Additional Components
1. Dashboard:
Utilizes Streamlit, HTML, and PowerBI.
2. Testing:
DeepCheck testing integrated.

## Roadmap
Dockerization and Automation with Jenkins (Coming Soon)

