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
    #Filling the missing values for continuous variables with mean
    df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].mean())
    df['MaxTemp']=df['MinTemp'].fillna(df['MaxTemp'].mean())
    df['Rainfall']=df['Rainfall'].fillna(df['Rainfall'].mean())
    df['Evaporation']=df['Evaporation'].fillna(df['Evaporation'].mean())
    df['Sunshine']=df['Sunshine'].fillna(df['Sunshine'].mean())
    df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
    df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean())
    df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean())
    df['Humidity9am']=df['Humidity9am'].fillna(df['Humidity9am'].mean())
    df['Humidity3pm']=df['Humidity3pm'].fillna(df['Humidity3pm'].mean())
    df['Pressure9am']=df['Pressure9am'].fillna(df['Pressure9am'].mean())
    df['Pressure3pm']=df['Pressure3pm'].fillna(df['Pressure3pm'].mean())
    df['Cloud9am']=df['Cloud9am'].fillna(df['Cloud9am'].mean())
    df['Cloud3pm']=df['Cloud3pm'].fillna(df['Cloud3pm'].mean())
    df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].mean())
    df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].mean())
    df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
    df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])

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