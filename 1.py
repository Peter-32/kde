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
X = housing_data[["_0"]]
print(X)


x = np.linspace(X.min(), X.max(), 1000)



kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
y = kde.score_samples(x)
sns.kdeplot(data=housing_data, x="_0")
plt.show()
# plt.plot(x, y)
# X.hist()
# plt.show()
#
#
# plt.show()
