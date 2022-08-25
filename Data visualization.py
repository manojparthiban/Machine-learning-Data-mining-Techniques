import numpy as np
import pandas as pd
df = pd.read_csv("Iris.csv")
n = np.array(df)
print("shape: ",n.shape)
print("\ndata:\n",df)
print("\nshape:",df.shape)
print("\ninfo:\n",df.info())
print("\nmean:\n",df.mean())
print("\nSepalLength MEAN :",df.loc[:,'sepal_length'].mean())
print("\nSepalWidth MEAN :",df.loc[:,'sepal_width'].mean()) 

#calculate the standard deviation of the first five rows
print("\nmean of first five rows:\n",df.mean(axis = 1)[0:5])
print("\nmode:\n",df.mode())
print("\nstandard deviation:\n",df.std())

print("\nstandard deviation of sepal_length :\n",df.loc[:,'sepal_length'].std())
print("\nstandard deviation of sepal width:\n",df.loc[:,'sepal_width'].std())

#calculate the standard deviation of the first five rows 
print("\nstandard deviation of first five rows:\n",df.std(axis = 1)[0:5])
print("\nVariance:\n",df.var())

from scipy.stats import iqr
print("\niqr: \n",iqr(df['sepal_length']))

print("\nskew:\n",df.skew())
print("\nDescibe:\n",df.describe())
print("\nDescribe all:\n",df.describe(include='all'))


import seaborn as sns 
import matplotlib.pyplot as plt
print("\nhead:\n",df.head())
sns.distplot(df['sepal_length'], kde =False).set_title("Histogram")
plt.show() 
sns.boxplot(df['sepal_length']).set_title("Boxplot")
plt.show()
sns.scatterplot(x=df.index,y=df["sepal_length"]).set_title("Scatterplot")
plt.show()
sns.lineplot(x=df.index,y=df["sepal_length"]).set_title("Lineplot")
plt.show()


import statsmodels.api as sm
# x versus quantiles/ppf of a distribution
#quantile-quantile plot for comparing two probability distributions by plotting their quantiles against each other
sm.qqplot(df["sepal_length"])
plt.show()

