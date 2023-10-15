import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
import dagshub
from data_processing import transform_data
# Load your raw data (assuming it's stored in a variable named df_raw)
# If you load from a file, adjust accordingly
df_raw = pd.read_csv('data/WeatherAUS.csv')

# Preprocess the data using your function
X_train_scaled, X_test_scaled, y_train, y_test = transform_data(df_raw)
# Initialize Dagshub project
dagshub.init("weather", "Sahar-dev", mlflow=True)
# Start MLflow run
mlflow.start_run()

# Log parameters
mlflow.log_param("dataset_size", len(df_raw))
mlflow.log_param("train_test_split_ratio", 0.2)

# Random Forest Regressor
with mlflow.start_run():
    random_forest_model = RandomForestRegressor(n_estimators=100, random_state=0)
    random_forest_model.fit(X_train_scaled, y_train)
    y_pred_random_forest = random_forest_model.predict(X_test_scaled)

    # Log parameters for Random Forest
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_params(random_forest_model.get_params())

    # Log metrics for Random Forest
    accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest.round())
    f1_random_forest = f1_score(y_test, y_pred_random_forest.round())
    mlflow.log_metric("accuracy", accuracy_random_forest)
    mlflow.log_metric("f1_score", f1_random_forest)

    # Log confusion matrix as an artifact
    conf_matrix_random_forest = confusion_matrix(y_test, y_pred_random_forest.round())
    mlflow.log_artifact(conf_matrix_random_forest, "confusion_matrix_random_forest.csv")

    # Save the Random Forest model
    mlflow.sklearn.log_model(random_forest_model, "RandomForestRegressor_model")

# Bernoulli Naive Bayes
with mlflow.start_run():
    model = BernoulliNB()
    model.fit(X_train_scaled, y_train)
    predicted = model.predict(X_test_scaled)

    # Log parameters for Bernoulli Naive Bayes
    mlflow.log_param("model", "BernoulliNB")
    mlflow.log_params(model.get_params())

    # Log metrics for Bernoulli Naive Bayes
    accuracy_naive_bayes = accuracy_score(y_test, predicted)
    f1_naive_bayes = f1_score(y_test, predicted)
    mlflow.log_metric("accuracy", accuracy_naive_bayes)
    mlflow.log_metric("f1_score", f1_naive_bayes)

    # Save the Bernoulli Naive Bayes model
    mlflow.sklearn.log_model(model, "BernoulliNB_model")

# LightGBM Regressor
with mlflow.start_run():
    model = lgb.LGBMRegressor(n_estimators=100, random_state=0)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Log parameters for LightGBM
    mlflow.log_param("model", "LGBMRegressor")
    mlflow.log_params(model.get_params())

    # Log metrics for LightGBM
    accuracy_lgbm = accuracy_score(y_test, (y_pred > 0.5).astype(int))
    f1_lgbm = f1_score(y_test, (y_pred > 0.5).astype(int))
    mlflow.log_metric("accuracy", accuracy_lgbm)
    mlflow.log_metric("f1_score", f1_lgbm)

    # Save the LightGBM model
    mlflow.sklearn.log_model(model, "LGBMRegressor_model")

# CatBoost Classifier
with mlflow.start_run():
    model = CatBoostClassifier(iterations=100, random_seed=0, logging_level='Silent')
    model.fit(X_train_scaled, y_train)
    y_pred_catboost = model.predict(X_test_scaled)

    # Log parameters for CatBoost
    mlflow.log_param("model", "CatBoostClassifier")
    mlflow.log_params(model.get_params())

    # Log metrics for CatBoost
    accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
    f1_catboost = f1_score(y_test, y_pred_catboost)
    mlflow.log_metric("accuracy", accuracy_catboost)
    mlflow.log_metric("f1_score", f1_catboost)

    # Save the CatBoost model
    mlflow.sklearn.log_model(model, "CatBoostClassifier_model")

# Isolation Forest
with mlflow.start_run():
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train_scaled)
    y_pred_isolation_forest = model.predict(X_test_scaled)

    # Convert predictions to binary values (0 or 1) based on anomalies
    y_pred_binary_isolation_forest = (y_pred_isolation_forest == -1).astype(int)

    # Log parameters for Isolation Forest
    mlflow.log_param("model", "IsolationForest")
    mlflow.log_params(model.get_params())

    # Log confusion matrix as an artifact for Isolation Forest
    conf_matrix_isolation_forest = confusion_matrix(y_test, y_pred_binary_isolation_forest)
    mlflow.log_artifact(conf_matrix_isolation_forest, "confusion_matrix_isolation_forest.csv")

    # Save the Isolation Forest model
    mlflow.sklearn.log_model(model, "IsolationForest_model")

# End MLflow run
mlflow.end_run()
