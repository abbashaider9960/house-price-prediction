import pandas as pd
import joblib
from pathlib import Path

# Saved model ka path
model_path = Path(__file__).resolve().parent / "house_price_model.pkl"

# Saved model load karo
loaded_model = joblib.load(model_path)

print("Model loaded successfully!")
print("Model type:", type(loaded_model).__name__)

income = float(input("Enter median income: "))
age = float(input("Enter median house age: "))
rooms = float(input("Enter average rooms: "))
bedrooms = float(input("Enter average bedrooms: "))
population = float(input("Enter population: "))
occupancy = float(input("Enter average occupancy: "))
latitude = float(input("Enter latitude: "))
longitude = float(input("Enter longitude: "))

new_house = pd.DataFrame([{
    "MedInc": income,
    "HouseAge": age,
    "AveRooms": rooms,
    "AveBedrms": bedrooms,
    "Population": population,
    "AveOccup": occupancy,
    "Latitude": latitude,
    "Longitude": longitude
}])

print(new_house.to_string(index=False))

prediction = loaded_model.predict(new_house)

price = prediction[0] * 100000

print("Predicted House Value: $", round(price, 2))
