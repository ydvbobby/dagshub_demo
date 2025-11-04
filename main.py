import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import root_mean_squared_error, r2_score


df = pd.read_csv('data/song.csv')

x = df.drop(['BeatsPerMinute',"id"],axis = 1)
y = df['BeatsPerMinute']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.05)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
n_neighbours = 5

mlflow.set_experiment("linear_regression")

with mlflow.start_run(run_name="run3"):
    

    lr = LinearRegression()
    scaler = StandardScaler()
    
    mlflow.sklearn.log_model(lr, "linear_regression")

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    lr.fit(x_train,y_train)

    predictions = lr.predict(x_test)
    
    rms = root_mean_squared_error(y_test, predictions)
    r2_score = r2_score(y_test, predictions)
    mlflow.log_metric('r2', r2_score)
    mlflow.log_metric('rms', rms)
    mlflow.log_artifact(__file__)
    
    
print(rms)
print(r2_score)
print(mlflow.get_tracking_uri())


