import pandas as pd
import pickle

def fix_missing_cols(training_cols, new_data):
    missing_cols = set(training_cols) - set(new_data.columns)
    # Add missing columns in the test set with default value equal to 0
    for c in missing_cols:
        new_data[c] = 0
    # Ensure the order of columns in the test set is the same as in the train set
    new_data = new_data[training_cols]
    return new_data

def clean_data_json(df):
    # replace 'No' and 'Yes' in RainToday with 0 and 1
    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    
    selected_features = ['Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud3pm', 'Temp3pm']
    df = df[selected_features]

    return df
