import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data = pd.DataFrame(housing.data)
housing_data.columns = ["_" + str(x) for x in housing_data.columns]
print(housing_data)

# Sample code
X = housing_data[["_0"]]
print(X)
x = np.linspace(X.min(), X.max(), 1000)
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
y = kde.score_samples(x)

# Plot code
# sns.kdeplot(data=housing_data, x="_0", y="_1", fill=True, thresh=0, levels=100, cmap="mako")
sns.kdeplot(data=housing_data, x="_0", y="_1", fill=True, cmap="Blues")
plt.show()
