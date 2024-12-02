import pandas as pd
import streamlit as st

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Betöltjük a California Housing adatokat
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

# Ellenőrizzük a hiányzó értékeket
print(data.isnull().sum())

# Normalizálás MinMaxScaler-rel (PRICE nélkül)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.drop('PRICE', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])  # Csak a bemeneti jellemzők

# A bemeneti (X) és célváltozó (y) kijelölése
X = scaled_data
y = data['PRICE']  # Eredeti célváltozó marad

# Adat felosztása tanító és teszt adatra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineáris regressziós modell létrehozása és betanítása
model = LinearRegression()
model.fit(X_train, y_train)

# Előrejelzés a teszt adatokon
y_pred = model.predict(X_test)

# Modell teljesítményének értékelése
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Streamlit alkalmazás
st.title("Házár predikció")
st.write("Add meg az ingatlan jellemzőit, és megjósoljuk az árát.")

# Input mezők az ingatlan jellemzőinek (minden szükséges jellemző)
medinc = st.number_input("Közepes jövedelem (MedInc)", min_value=0.0, max_value=20.0, value=3.0)
houseage = st.number_input("Ház életkora (HouseAge)", min_value=1.0, max_value=100.0, value=20.0)
ave_rooms = st.number_input("Átlagos szobaszám (AveRooms)", min_value=1.0, max_value=10.0, value=5.0)
ave_bedrooms = st.number_input("Átlagos hálószobaszám (AveBedrms)", min_value=1.0, max_value=5.0, value=2.0)