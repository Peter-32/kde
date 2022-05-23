import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns

def time_convert(name, sec):
    mins = sec // 60
    sec = sec % 60
    hours = mins // 60
    mins = mins % 60
    print(f"{name} - Time Lapsed = {int(hours)}:{int(mins)}:{sec}")

def start_stopwatch():
    times.append(time.time())

def check_stopwatch(name):
    times.append(time.time())
    time_convert(name, times[-1] - times[-2])

times = []



housing = fetch_california_housing()
m, n = housing.data.shape
housing_data = pd.DataFrame(housing.data)
housing_data.columns = ["_" + str(x) for x in housing_data.columns]

# Sample code
X = housing_data[["_0"]]
X = X.iloc[0:1000]
start_stopwatch()
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
kde.sample()
check_stopwatch("1,000 rows")
print("1,000 -", X.info())

X = X.sample(n=10000, replace=True)
start_stopwatch()
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
kde.sample()
check_stopwatch("10,000 rows")
print("10,000 -", X.info())
print(X.info())

X = X.sample(n=100000, replace=True)
start_stopwatch()
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
kde.sample()
check_stopwatch("100,000 rows")
print(X.info())

X = X.sample(n=1000000, replace=True)
start_stopwatch()
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
kde.sample()
check_stopwatch("1,000,000 rows")
print(X.info())

# Anything larger should be sampled to 1 million rows
# df.sample(n=1000000, replace=False)
