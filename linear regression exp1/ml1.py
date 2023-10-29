import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data = pd.read_csv('BostonHousing.csv')

linr = LinearRegression()

data['medv'] = np.log1p(data['medv'])
co = data.corr()
print(co)
x = data.drop(['medv','b'], axis=1)
y = data['medv']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.3)
linr.fit(x_train, y_train)
y_pred = linr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Square Error:", mse)
print("Mean Absolute Error:", mae)

