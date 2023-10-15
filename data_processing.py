import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def transform_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df.drop(['Date'], axis = 1,inplace=True) 
    df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
    df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
    selected_features = ['Rainfall', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud3pm', 'Temp3pm']
    features = df[selected_features]
    target = df['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42,
                                                    shuffle=True, stratify=target)
                                                    # Normalize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled,X_test_scaled,y_train,y_test