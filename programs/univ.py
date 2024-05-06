import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('C:\\Users\\Johanan\\OneDrive\\Documents\\GitHub\\dynamic-pricing\\data\\HotelBookingDataset.csv')

# Univariate analysis for each attribute
for column in df.columns:
    if df[column].dtype == 'object':
        # For categorical variables
        counts = df[column].value_counts()
        print(f"\n{column} distribution:")
        print(counts)
        plt.bar(counts.index, counts.values)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"{column} distribution")
        plt.xticks(rotation=45)
        plt.show()
    else:
        # For numerical variables
        print(f"\n{column} summary statistics:")
        print(df[column].describe())
        plt.hist(df[column], bins=10)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"{column} distribution")
        plt.show()

