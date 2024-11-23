import pandas as pd
import numpy as np

# Load data
red_wine_data = pd.read_csv('data\winequality-red.csv', delimiter=';')
white_wine_data = pd.read_csv('data\winequality-white.csv', delimiter=';')
combined_data = pd.concat([red_wine_data, white_wine_data], axis=0)
combined_data.reset_index(drop=True, inplace=True)
combined_data.drop('quality', axis=1, inplace=True)

# Initialize lists to store means and standard deviations
means = []
std_devs = []
columns = combined_data.columns

# Iterate over each column in the DataFrame
for column in combined_data.columns:
    means.append(combined_data[column].mean())
    std_devs.append(combined_data[column].std())

def generate_synthetic_data_point(column_name):
    if column_name not in columns:
        raise ValueError("Column name not found")
    column_index = columns.get_loc(column_name)
    synthetic_data_point = round(np.random.normal(loc=means[column_index], scale=std_devs[column_index]), 1)
    return synthetic_data_point



