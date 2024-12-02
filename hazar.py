import pandas as pd

from sklearn.datasets import fetch_california_housing

# Betöltjük a California Housing adatokat
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target