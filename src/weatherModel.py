from pydantic import BaseModel

class TransactionModel(BaseModel):
    Date: str
    Rainfall: float
    Sunshine: float
    WindGustSpeed: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud3pm: float
    Temp3pm: float
    RainToday: str
    RainTomorrow: str
