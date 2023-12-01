import pandas as pd
import pickle
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def fix_missing_cols(training_cols, new_data):
    missing_cols = set(training_cols) - set(new_data.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        new_data[c] = 0
    # Ensure the order of column in the test set is in the same order as in the train set
    new_data = new_data[training_cols]
    return new_data

def clean_data_json(df):
    col_todate = ["trans_date_trans_time", "dob"]
    # transform specific cols to datetime type
    for col in col_todate:
        # convert trans_date_trans_time, dob to datetime
        df[col] = pd.to_datetime(df[col])
    # extract new cols
    # create new columns day, month, year
    df["year"] = df["trans_date_trans_time"].dt.year
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day"] = df["trans_date_trans_time"].dt.day
    # Extract hour, minute, and second
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["month"] = df["trans_date_trans_time"].dt.month 
    df["sec"] = df["trans_date_trans_time"].dt.second
    # Extract age of cardholder column
    df['age'] = dt.date.today().year - pd.to_datetime(df['dob']).dt.year
    # drop unuseful columns
    df.drop(["dob", "trans_date_trans_time"], axis=1, inplace=True)
    # select numerical features
    num_features = df.select_dtypes(include=['integer']).columns.tolist()
    # select categorical features
    categ_features = df.select_dtypes(include=['object']).columns.tolist()
    encode_dict = {  # Encoding dictionary
        'F': 0, 'M': 1}
    df['gender'] = df['gender'].map(encode_dict)
    dummy_cols = ['category']
    df = pd.get_dummies(df, columns=dummy_cols)
    with open('training_cols.pkl', 'rb') as f:
        training_cols = pickle.load(f)  
    df = fix_missing_cols(training_cols, df)
    return df

def transform_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df.drop(['Date'], axis=1, inplace=True) 
    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
    # Filling the missing values for continuous variables with mean
    continuous_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
                       'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                       'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    for col in continuous_cols:
        df[col] = df[col].fillna(df[col].mean())

    selected_features = ['Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am',
                         'Pressure3pm', 'Cloud3pm', 'Temp3pm']
    features = df[selected_features]
    target = df['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42,
                                                        shuffle=True, stratify=target)
    # Normalize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
