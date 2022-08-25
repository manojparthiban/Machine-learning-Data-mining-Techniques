import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def cov(c,d):
    covar = 0
    for k in range(2,len(c)): 
        covar += c[k]*d[k]
    return covar/len(c)
    
#Data set 
df=pd.read_csv("world_population.csv")
print("\nData set:\n\n",df)

#Standardizing data set
col = df.columns.tolist()
for i in range(1,len(col)):
    c = df[col[i]]
    for j in range( 0,len(c)):
        c[j] = round((c[j]-c.mean())/c.std(),2)
print("\nStandardized data set:\n\n",df)

#Finding Covariance matrix
for i in range(1,len(col)):
    c = df[col[i]]
    for j in range(0,len(c)):
        if i-1==j: 
            c[j] = c.var()
        else:
            d = df[col[j+1]]
            c[j] = cov(c,d)
df = round(df,2)
print("\nCovariance Matrix:\n\n",df)

#Finding EigenVectors
df.pop("Country/Other")
eigenvalues,eigenvectors = np.linalg.eig(df)
df = round(df*abs(eigenvectors),2)
print("\nTransformed data set:\n\n",df)

#Plotting principal components
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("Scatter plot pf principal components")
plt.scatter(x = df["Yearly Change"],y = df["Net Change"])
