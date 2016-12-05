#! /usr/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("train.csv")
df.drop("Id", axis=1, inplace=True)
data = pd.get_dummies(df)
X, Y = data.iloc[:,1:].values, data["Score"].values
x = np.zeros(70, np.int32)
for i, v in Counter(Y).items():
    x[i] = v
print(np.sum(x))
print(x)
plt.bar(np.arange(0, 70, 1), x)
plt.show()

z = np.array(x, np.float32)
z /= int(np.sum(x))
t = 0.0
for i in range(z.shape[0]):
    if i > 0:
        t += z[i]
        z[i] = t
print(z)
plt.plot(np.arange(0, 70, 1), z)
plt.show()
