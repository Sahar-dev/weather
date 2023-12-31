import streamlit as st
import requests

def input_form():
    """
    Streamlit input form for weather prediction.

    Returns:
    - tuple: Tuple containing user input values.
    """
    st.title("Weather Prediction App")

    # Input Form
    location = st.text_input("Enter location:")
    date = st.date_input("Select date", value=None, min_value=None, max_value=None, key=None)
    min_temp = st.number_input("Min Temperature", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    max_temp = st.number_input("Max Temperature", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    rainfall = st.number_input("Rainfall", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    evaporation = st.number_input("Evaporation", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    sunshine = st.number_input("Sunshine", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    wind_gust_dir = st.text_input("Wind Gust Direction:")
    wind_gust_speed = st.number_input("Wind Gust Speed", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    wind_dir_9am = st.text_input("Wind Direction 9am:")
    wind_dir_3pm = st.text_input("Wind Direction 3pm:")
    wind_speed_9am = st.number_input("Wind Speed 9am", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    wind_speed_3pm = st.number_input("Wind Speed 3pm", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    humidity_9am = st.number_input("Humidity 9am", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    humidity_3pm = st.number_input("Humidity 3pm", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    pressure_9am = st.number_input("Pressure 9am", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    pressure_3pm = st.number_input("Pressure 3pm", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    cloud_9am = st.number_input("Cloud 9am", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    cloud_3pm = st.number_input("Cloud 3pm", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    temp_9am = st.number_input("Temperature 9am", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    temp_3pm = st.number_input("Temperature 3pm", min_value=None, max_value=None, value=0.0, step=None, format=None, key=None)
    rain_today = st.selectbox("Rain Today", options=["Yes", "No"])

    return location, date, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_dir, wind_gust_speed, \
           wind_dir_9am, wind_dir_3pm, wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, \
           pressure_3pm, cloud_9am, cloud_3pm, temp_9am, temp_3pm, rain_today

def main():
    """
    Main function for the Streamlit app.
    """
    location, date, min_temp, max_temp, rainfall, evaporation, sunshine, wind_gust_dir, wind_gust_speed, \
           wind_dir_9am, wind_dir_3pm, wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, \
           pressure_3pm, cloud_9am, cloud_3pm, temp_9am, temp_3pm, rain_today = input_form()

    if st.button("Predict Weather"):
        # Make a request to FastAPI backend
        request_data = {
            "Date": str(date),
            "Location": str(location),
            "MinTemp": float(min_temp),
            "MaxTemp": float(max_temp),
            "Rainfall": float(rainfall),
            "Evaporation": float(evaporation),
            "Sunshine": float(sunshine),
            "WindGustDir": str(wind_gust_dir),
            "WindGustSpeed": float(wind_gust_speed),
            "WindDir9am": str(wind_dir_9am),
            "WindDir3pm": str(wind_dir_3pm),
            "WindSpeed9am": float(wind_speed_9am),
            "WindSpeed3pm": float(wind_speed_3pm),
            "Humidity9am": float(humidity_9am),
            "Humidity3pm": float(humidity_3pm),
            "Pressure9am": float(pressure_9am),
            "Pressure3pm": float(pressure_3pm),
            "Cloud9am": float(cloud_9am),
            "Cloud3pm": float(cloud_3pm),
            "Temp9am": float(temp_9am),
            "Temp3pm": float(temp_3pm),
            "RainToday": str(rain_today),
        }

        response = requests.post("https://weather-app-prediction.onrender.com/predict", json=request_data)
        predictions = response.json().get("predictions", [])

        if predictions == [0]:
            st.text("You are safe, no rain tomorrow! Enjoy your day 😃")
        else:
            st.text("Warning! It's going to rain! Get your umbrella ☂️ ready!")

        # Display predictions
        st.write(f"Predicted Weather: {predictions}")

if __name__ == "__main__":
    main()
