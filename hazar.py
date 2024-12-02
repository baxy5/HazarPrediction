import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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