import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('C:\\Users\\Johanan\\OneDrive\\Documents\\GitHub\\dynamic-pricing\\data\\HotelBookingDataset.csv')  

# Bivariate analysis for selected pairs of attributes
pairs = [('lead_time', 'adr'), ('total_people', 'adr'), ('room_type', 'adr'), ('local_events', 'adr'), ('length_of_stay', 'adr')]

for pair in pairs:
    attribute1, attribute2 = pair
    
    if df[attribute1].dtype == 'object' and df[attribute2].dtype == 'object':
        # For two categorical variables
        cross_tab = pd.crosstab(df[attribute1], df[attribute2])
        sns.heatmap(cross_tab, annot=True, fmt="d")
        plt.xlabel(attribute2)
        plt.ylabel(attribute1)
        plt.title(f"Cross-tabulation between {attribute1} and {attribute2}")
        plt.show()
    elif df[attribute1].dtype != 'object' and df[attribute2].dtype != 'object':
        # For two numerical variables
        sns.scatterplot(data=df, x=attribute1, y=attribute2)
        plt.xlabel(attribute1)
        plt.ylabel(attribute2)
        plt.title(f"Scatter plot between {attribute1} and {attribute2}")
        plt.show()
    else:
        # For one categorical and one numerical variable
        sns.boxplot(data=df, x=attribute1, y=attribute2)
        plt.xlabel(attribute1)
        plt.ylabel(attribute2)
        plt.title(f"Box plot of {attribute2} by {attribute1}")
        plt.xticks(rotation=45)
        plt.show()
