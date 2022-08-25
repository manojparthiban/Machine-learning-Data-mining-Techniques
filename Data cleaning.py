import pandas as pd
from scipy.stats import iqr

df=pd.read_csv("world_population.csv")
print("Data Frame:\n",df)

# 1) Replace missing data
df.fillna(0,inplace=True)
print("\n\nReplaced missing values: \n",df)

# 2) Remove Irrelevant data
column_dat = df["World Share"].tolist()
for i in range(0,len(column_dat)):
    if (column_dat[i].isalpha()):
        df["World Share"] = df["World Share"].replace(column_dat[i],0)
print("\n\nRemoved irrelevant data:\n",df)

# 3) Rename wrong attributes names
df.rename(columns = {'Densit':'Density'}, inplace = True)
print("\n\nRenamed wrong column names:\n",list(df.columns))

# 4) Remove duplicate rows
df.drop_duplicates(inplace=True)
print("\n\nRemoved duplicate row:\n",df)

df.to_csv("world_population.csv",index=False)
df = pd.read_csv("world_population.csv")

# 5) Remove outliers
column_dat = df["Yearly Change"].tolist()
outlier_range = (iqr(column_dat)*1.5)
for i in range(0,len(column_dat)):
    if (outlier_range<column_dat[i]):
        df["Yearly Change"] = df["Yearly Change"].replace(column_dat[i],0)
print("\n\nRemoving outliers:\n",df)
