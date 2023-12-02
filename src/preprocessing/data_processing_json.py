import pandas as pd
import pickle

def fix_missing_cols(training_cols, new_data):
    """
    Fix missing columns in the new data based on the training columns.

    Parameters:
    - training_cols (list): List of columns from the training set.
    - new_data (pd.DataFrame): New data to be fixed.

    Returns:
    - pd.DataFrame: New data with missing columns added and in the same order as training columns.
    """
    missing_cols = set(training_cols) - set(new_data.columns)

    # Add missing columns in the new data with default value equal to 0
    for c in missing_cols:
        new_data[c] = 0

    # Ensure the order of columns in the new data is the same as in the training set
    new_data = new_data[training_cols]
    return new_data

def clean_data_json(df):
    """
    Clean the input DataFrame by replacing values and selecting features.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Cleaned DataFrame with selected features.
    """
    # Replace 'No' and 'Yes' in RainToday with 0 and 1
    df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
    
    # Select specific features
    selected_features = ['Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud3pm', 'Temp3pm']
    df = df[selected_features]

    return df
