# Import libraries
import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from spellchecker import SpellChecker
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

# Function to load data from Excel file
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to save DataFrame to Excel file
def save_data(df, file_path):
    df.to_excel(file_path, index=False)

def drop_script(df):
    # Drop rows where the 'Script Type' column has empty cells
    df = df.dropna(subset=['Script Type'])
    return df

# Main function to execute the entire data processing pipeline
def main():
    file_path = '/home/tofi-machine/Documents/DataMining/DataMining/test_sample.xlsx'
    save_folder = '/home/tofi-machine/Documents/DataMining/DataMining'
    df = load_data(file_path)

    # Additional processing steps...

    # Drop rows with empty cells in the 'Script Type' column
    df = drop_script(df)

    # Additional processing steps...

    save_data(df, file_path)
    print("Excel file updated")

if __name__ == "__main__":
    main()