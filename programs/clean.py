import pandas as pd
import numpy as np

#1 handled missing vales
path=r'D:\faqbot\dynamic-pricing\data\5000sample.csv'
df = pd.read_csv(path)
"""df.fillna(0, inplace=True)
print(df.isnull().sum())"""
# after wards df.to_csv('cleaned.csv', index=False)
#------
#2.handling duplicate
"""print("Original DataFrame:")
print(df.head())
duply=df[df.duplicated()]
if not duply.empty:
    df.drop_duplicates(inplace=True)
    print("\nDuplicate rows removed.")
print("\nDataFrame After Handling Duplicates:")
print(df.head())"""
#3outliers
"""minimumprice = df.groupby('Room Type')['Room Price ($)'].min()
meanprice = df.groupby('Room Type')['Room Price ($)'].mean().round()
maxprice=df.groupby('Room Type')['Room Price ($)'].max()
print("Minimum price of both:",minimumprice)
print("Mean price of both:", meanprice)
print("Maximum price of both:",maxprice)"""
z_scores = (df['Room Price ($)'] - df['Room Price ($)'].mean()) / df['Room Price ($)'].std()
df2= df[np.abs(z_scores) <= 3] 
print(df2.head())