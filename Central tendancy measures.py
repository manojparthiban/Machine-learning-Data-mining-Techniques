# statistical analysis = mean, mode, median, standard deviation, iqr, variance, skew, plots

import numpy as np
import pandas as pd
df = pd.read_csv("world_population.csv")
n = np.array(df)
print("\ndata:\n",df)
print("\nshape:\n",df.shape)
print("\ninfo:\n",df.info())

print("\nmean:\n",df.mean())
print("\nPopulation MEAN :",df.loc[:,'Population'].mean())
print("\nmode:\n",df.mode())

print("\nstandard deviation:\n",df.std())
print("\nstandard deviation of Population :",df.loc[:,'Population'].std())

print("\nVariance:\n",df.var())

from scipy.stats import iqr
print("\niqr: \n",iqr(df['Population']))

print("\nskew:\n",df.skew())
print("\nDescibe:\n",df.describe())
print("\nDescribe all:\n",df.describe(include='all'))

import seaborn as sns 
import matplotlib.pyplot as plt
print("\nhead:\n",df.head())
sns.distplot(df['Population'], kde =False).set_title("Histogram")
plt.show() 
sns.boxplot(df['Population']).set_title("Boxplot")
plt.show()
sns.scatterplot(x=df.index,y=df["Population"]).set_title("Scatterplot")
plt.show()
sns.lineplot(x=df.index,y=df["Population"]).set_title("Lineplot")
plt.show()
