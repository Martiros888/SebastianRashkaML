import numpy as np
import pandas as pd
import seaborn as sns
from math import pi
from matplotlib.pyplot import plot

arr = np.arange(-200, 200)
print(arr)

def f(x):
    return (9 - (x ** 2)) 




def generateY(arr):
    array = np.array([])
    for i in arr:
        array = np.array([*array, f(i)])
    return array

df = pd.DataFrame(columns=['X', 'y'])
df['X'] = arr
df['y'] = generateY(arr)

X = df["X"].values
y = df['y'].values

plot(X, y)
print(y.max())