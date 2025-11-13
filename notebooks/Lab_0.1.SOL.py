import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# create a series with a list
s = pd.Series([7,"Hello World!",42.46], index=['A','B','C'])
print("My series:")
print(s)


# create a series from a dictionary
d = {1:"ciao",2:"miao",3:"bau"}
sfd = pd.Series(d)
print(sfd)

dd = {'Chicago':1000, 'New York':1300, 'Portland':900, 'San Francisco':1100,
     'Austin':450, 'Boston':None}
cities = pd.Series(dd)
print("\nCities Series:")
print(cities)

print(cities[cities>1000])


s1 = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
s2 = pd.Series([4, 5, 6], index=['A', 'D', 'C'])

print("\ns1: ")
print(s1)
print("\ns2: ")
print(s2)
print("\ns1 + s2: ")
ss = s1 + s2
print(ss)


ignPath = "https://raw.githubusercontent.com/nickplas/Intro_to_ML_24-25/main/data/ign.csv"
ignDataset = pd.read_csv(ignPath)
print(ignDataset.head())

ignDataset = ignDataset.iloc[:,1:]
print(ignDataset.head())