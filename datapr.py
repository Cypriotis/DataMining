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

#from datacheck  import standardize_text

# Function to load data from Excel file
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Function to save DataFrame to Excel file
def save_data(df, file_path):
    df.to_excel(file_path, index=False)

# Function to log the current date and time in red
def log_datetime():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
    red_text = '\033[91m'
    reset_text_color = '\033[0m'
    print(f"{red_text}{formatted_datetime}{reset_text_color}")

def drop_duplicates(df):
    duplicates = df[df.duplicated(subset="Film", keep=False)]
    if not duplicates.empty:
        print("Duplicate rows in the 'Film' column:")
        print(str(duplicates))
        df = df.drop(duplicates.index)
        print("Duplicate rows in the 'Film' column have been dropped.")
    else:
        print("No duplicates found in the 'Film' column.")
    return df

def drop_columns(df):
    list_of_columns_to_delete  = ['Film', 'Rotten Tomatoes vs Metacritic  deviance', 'Primary Genre', 'Opening Weekend',
                                'Opening weekend ($million)', ' Budget recovered', ' Budget recovered opening weekend',' of Gross earned abroad','Release Date (US)', 'Distributor', 'Oscar Detail']
    columns_exist = all(col in df.columns for col in list_of_columns_to_delete)

    if columns_exist:
        # Delete the specified columns
        df = df.drop(columns=list_of_columns_to_delete, axis=1)
    return df

def fill_oscar(df):
    df['Oscar Winners'] = df['Oscar Winners'].fillna('not')
    return df

def fill_zero(df):
    df['IMDb Rating'] = df['IMDb Rating'].fillna(0)
    df['IMDB vs RT disparity'] = df['IMDB vs RT disparity'].fillna(0)
    return df

def fill_nan(df):
    df = df.fillna(0)
    return df

import pandas as pd

def convert_non_numeric_columns(df):
    # Specify the columns to convert to numeric
    columns_to_convert = [
        'Domestic gross ($million)',
        'Foreign Gross ($million)',
        'Worldwide Gross ($million)',
        'Budget ($million)'
    ]
    
    # Reset index of df
    df = df.reset_index(drop=True)
    
    try:
        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns_to_convert if col not in df.columns]
        if missing_columns:
            print(f"Columns not found in the DataFrame: {missing_columns}")
            return df  # Return the original DataFrame if columns are missing

        # Use apply with a lambda function to convert each element to numeric for specified columns
        for column_name in columns_to_convert:
            # Convert to numeric and handle NaN values
            df[column_name] = pd.to_numeric(df[column_name].replace(',', '.'), errors='coerce').fillna(0).astype(float)
        # Use apply with a lambda function to convert each element to numeric for specified columns
        for column_name in columns_to_convert:
            # Convert to numeric and handle NaN values
            df[column_name] = df[column_name].apply(lambda x: round(pd.to_numeric(str(x).replace(',', '.'), errors='coerce')) if pd.notna(x) else x)


        return df
    
    except Exception as e:
        print(f"Error converting columns to numeric: {e}")
        return df
    
def handle_minus_symbol(df):
    #a function to drop any rows that contain a '-' symbol in any cell of the dataframe
    df = df[~df.isin(['-'])]
    return df

def drop_empty(df):
    #drop nan or empty cells in dataframe df
    df = df.dropna()
    return df

def standardize(df):
    #standardize text on a column to titlecase
    df['Oscar Winners'] = df['Oscar Winners'].str.title()
    return df

# Function to correct spelling in a given cell using SpellChecker
def correct_spelling(cell_content, spell_checker):
    if pd.notna(cell_content):
        words = [word.strip() for word in cell_content.split(',')]
        corrected_words = [spell_checker.correction(word) if spell_checker.correction(word) else word for word in words]
        return ', '.join(corrected_words)
    else:
        return cell_content

# Function to apply spell checker to a specified column
def apply_spell_checker(df, column_name):
    spell = SpellChecker()
    df[column_name] = df[column_name].apply(lambda x: correct_spelling(x, spell))
    print("Spell checher success")
    return df

# Function to remove anything after a comma in a specified column
def remove_after_comma(cell_content):
    if pd.notna(cell_content):
        # Check the data type before splitting
        if isinstance(cell_content, str):
            return cell_content.split(',')[0]
        else:
            return cell_content
    else:
        return cell_content

def remove_text_after_comma(df, column_name):
    df[column_name] = df[column_name].apply(remove_after_comma)
    print("removed commas")
    return df  # Add this line to return the modified DataFrame
    








# Main function to execute the entire data processing pipeline
def main():
    file_path = '/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx'
    save_folder = '/home/tofi-machine/Documents/DataMining/DataMining'
    df = load_data(file_path)
    log_datetime()

    #Drop Duplicate Values
    df = drop_duplicates(df)

    #Drop irrelevant Columns
    df = drop_columns(df)

    #Fill nan Oscar Winners column cells with not
    df = fill_oscar(df)

    #reset index of df
    df.reset_index(drop=True, inplace=True)

    #Fill IMDb Rating column and IMDB vs RT disparity column with 0
    df = fill_zero(df)

    #Fill nan cells with 0   //I will try to drop the rows instead, based on the prediction results
    df = fill_nan(df)

    #reset index of df
    df.reset_index(drop=True, inplace=True)

    #Convert non-numeric columns to numeric
    df = convert_non_numeric_columns(df)

    #Drop rows that contain a '-' symbol in any cell
    df = handle_minus_symbol(df)

    #Drop any rows with empty cells
    df = drop_empty(df)

    #standarise text on a column to titlecase
    df = standardize(df)


    #Remove text after comma
    df = remove_text_after_comma(df, 'Genre')


    

    save_data(df, file_path)
    print("Excel file updated")

if __name__ == "__main__":
    main()