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


desired_formatting = 'titlecase'  # Replace with your desired formatting

# Function to standardize text
def standardize_text(text):
    if desired_formatting == 'uppercase':
        return text.group(0).upper()
    elif desired_formatting == 'lowercase':
        return text.group(0).lower()
    elif desired_formatting == 'titlecase':
        return text.group(0).title()
    else:
        return text.group(0)  # No formatting specified

def clean_and_convert_to_numeric(value):
    if isinstance(value, (int, float)):
        # If the value is already numeric, return it
        return value
    elif isinstance(value, str):
        # If the value is a string and not '-', remove commas and convert to float
        if value != '-':
            return float(value.replace(',', ''))
        else:
            return pd.NaT  # Replace '-' with NaN
    else:
        # For other types, return NaN or handle as appropriate
        return pd.NaT  # Use pd.NaT for missing/undefined values
def convert_columns_to_numeric(df, columns_to_convert):
    """
    Convert specified columns in the DataFrame to integer.

    Parameters:
    - df (pd.DataFrame): DataFrame.
    - columns_to_convert (list): List of column names to convert to integer.

    Returns:
    - df_integer (pd.DataFrame): DataFrame with the specified columns converted to integer.
    """
    # Reset index of df
    df = df.reset_index(drop=True)
    try:
        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns_to_convert if col not in df.columns]
        if missing_columns:
            print(f"Columns not found in the DataFrame: {missing_columns}")
            return None

        # Use apply with a lambda function to convert each element to numeric for specified columns
        for column_name in columns_to_convert:
            # Convert to numeric and handle NaN values
            df[column_name] = df[column_name].apply(lambda x: round(pd.to_numeric(str(x).replace(',', '.'), errors='coerce')) if pd.notna(x) else x)

        return df
    except Exception as e:
        print(f"Error converting columns to integer: {e}")
        return None


# Function to fill missing values in the 'Oscar Winners' column
def fill_missing_oscar_values(df):
    # Print the columns before filling missing values
    print("Columns before filling missing values:", df.columns)

    # If 'one-hot encoding Oscar Winners' column is present, no need to fill NaN values
    #df['Script Type'] = df['Script Type'].fillna("")

    # Print the columns after filling missing values
    print("Columns after filling missing values:", df.columns)

    return df

# Function to drop duplicate rows based on the 'Film' column
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

# Function to drop irrelevant columns from PART1
def drop_irrelevant_columns(df, columns_to_delete):
    if all(col in df.columns for col in columns_to_delete):
        df = df.drop(columns=columns_to_delete, axis=1)
    return df


# Function to standardize text in the 'Oscar Winners' column
def standardize_text_column(df):
    # Define the phrase you want to find and standardize
    target_phrase = 'oscar winner'  # Replace with the phrase you want to find

    # Define the desired formatting (e.g., 'uppercase', 'lowercase', 'titlecase')

    # Convert the column to strings
    df['Oscar Winners'] = df['Oscar Winners'].astype(str)

    # Apply the standardization function to the column containing the target phrase
    df['Oscar Winners'] = df['Oscar Winners'].apply(lambda x: re.sub(target_phrase, standardize_text, x, flags=re.IGNORECASE))
    return df

# Function to calculate the percentage of missing values in each column
def calculate_missing_percent(df):
    return (df.isnull().sum() / len(df)) * 100

# Function to delete columns with more than 2.5% missing values
def delete_columns_with_missing_values(df, columns_to_delete):
    df.drop(columns=columns_to_delete, inplace=True)
    return df

# Function to drop rows with any empty cells
def drop_rows_with_empty_cells(df):
    df_cleaned = df.dropna()
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

# Function to check and delete rows where the minimum value in specified columns is zero
def check_min_value_zero(df, numeric_columns):
    columns_with_min_zero = [column for column in numeric_columns if df[column].min() == 0]
    for column in columns_with_min_zero:
        df = df[df[column] != 0]
    return df

# Function to mark and delete rows with '-' symbol without a number in any cell
def contains_minus_without_number(row):
    return any('-' in str(cell) and not any(char.isdigit() for char in str(cell).split('-')[1]) for cell in row)

# Function to delete rows marked for deletion based on the condition
def delete_rows_with_minus_symbol(df):
    rows_to_delete = df.apply(contains_minus_without_number, axis=1)
    df = df[~rows_to_delete]
    df_clean = df.dropna()
    return df_clean


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
        return cell_content.split(',')[0]
    else:
        return cell_content
    

# Function to apply removing anything after a comma to a specified column
def remove_text_after_comma(df, column_name):
    df[column_name] = df[column_name].apply(remove_after_comma)
    print("removed commas")

    return df

# Function to invert the values in a specified column
def invert_column_values(df, column_name_to_invert):
    df[column_name_to_invert] = df[column_name_to_invert].apply(lambda x: 1 if x == 0 else 0)
    return df

# Function to apply random deletion based on a deletion probability to a specified column
def random_deletion(df):
    # Assuming 'df' is your DataFrame and 'column_name' is the column you want to check
    text_to_remove = 'not Oscar Winners'
    removal_percentage = 0.9  # Replace with the desired percentage

    # Create a boolean mask for rows that contain the specified text (case-insensitive and substring check)
    mask = df['Oscar Winners'].str.contains(text_to_remove, case=False, na=False)

    # Identify the number of rows to remove based on the percentage
    rows_to_remove = int(len(df) * removal_percentage)

    # Get the indices of the rows to remove
    indices_to_remove = df[mask].sample(rows_to_remove).index

    # Remove the identified rows
    df = df.drop(indices_to_remove)

    # Reset the index if needed
    df.reset_index(drop=True, inplace=True)
    save_data(df, '/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx')

# Function to drop irrelevant columns from PART3
def drop_irrelevant_columns(df, columns_to_delete):
    # Check if the specified columns exist in the DataFrame
    columns_exist = all(col in df.columns for col in columns_to_delete)

    if columns_exist:
        # Delete the specified columns
        df = df.drop(columns=columns_to_delete)
    return df

def detect_and_drop_minus_rows(df):
    # Function to check if a cell contains only the '-' symbol
    def contains_only_minus(cell):
        return str(cell).strip() == '-'

    # Apply the function to each row
    rows_to_drop = df.apply(lambda row: all(contains_only_minus(cell) for cell in row), axis=1)

    # Drop the rows where all cells contain only the '-' symbol
    df_cleaned = df[~rows_to_drop]

    # Reset the index
    df_cleaned.reset_index(drop=True, inplace=True)

    return df_cleaned

def detect_and_drop_nan_rows(df):
    """
    Detect and drop rows with NaN values.

    Parameters:
    - df (pd.DataFrame): DataFrame.

    Returns:
    - df_cleaned (pd.DataFrame): DataFrame with rows containing NaN values removed.
    """
    # Detect rows with NaN values
    rows_with_nan = df[df.isnull().any(axis=1)]

    if not rows_with_nan.empty:
        print("Rows with NaN values detected:")
        print(rows_with_nan)

        # Drop rows with NaN values
        df_cleaned = df.dropna()

        # Reset the index
        df_cleaned.reset_index(drop=True, inplace=True)

        print("Rows with NaN values have been dropped.")
    else:
        print("No rows with NaN values found.")
        df_cleaned = df.copy()

    #reset df index
    df_cleaned.reset_index(drop=True, inplace=True)

    return df_cleaned

# Function to display basic information about the dataset
def display_basic_info(df):
    print("Number of rows and columns:", df.shape)
    print("Column names:", df.columns)
    print("Data types of columns:")
    print(df.dtypes)

# Function to analyze Oscar winners and print results
def analyze_oscar_winners(df):
    total_records = len(df)
    oscar_winners = df[df['one-hot encoding Oscar Winners'] == 1]
    num_oscar_winners = len(oscar_winners)
    percentage_oscar_winners = (num_oscar_winners / total_records) * 100
    percentage_non_oscar_winners = 100 - percentage_oscar_winners

    print(f"Total records: {total_records}")
    print(f"Number of Oscar winners: {num_oscar_winners}")
    print(f"Percentage of Oscar winners: {percentage_oscar_winners:.2f}%")
    print(f"Percentage of non-Oscar winners: {percentage_non_oscar_winners:.2f}%")

# Function to save a pie chart showing the percentage of Oscar winners
def save_pie_chart(df, save_folder, total_records):
    percentage_oscar_winners = (df['one-hot encoding Oscar Winners'].sum() / total_records) * 100
    percentage_non_oscar_winners = 100 - percentage_oscar_winners

    fig = px.pie(
        names=['Oscar Winners', 'Non-Oscar Winners'],
        values=[percentage_oscar_winners, percentage_non_oscar_winners],
        title=f"Percentage of Oscar Winners in the Dataset ({total_records} records)"
    )

    save_filename = 'oscar_winners_pie_chart.png'
    save_path = os.path.join(save_folder, save_filename)
    fig.write_image(save_path, format='png')

# Function to plot a correlation heatmap for numeric columns
def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=['number'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# Function to plot a pairplot for selected numeric columns
def plot_pairplot(df):
    selected_columns = ['Budget ($million)', 'Domestic gross ($million)', 'Foreign Gross ($million)', 'Worldwide Gross ($million)']
    sns.pairplot(df[selected_columns])
    plt.title("Pairplot of Selected Numeric Columns")
    plt.show()

# Function to plot a countplot for Oscar Winners
def plot_countplot(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='one-hot encoding Oscar Winners')
    plt.title("Count of Oscar Winners")
    plt.show()

# Function to plot histograms for numeric columns
def plot_histogram(df, numeric_columns):
    for column in numeric_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

# Main function to execute the entire data processing pipeline
def main():
    file_path = '/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx'
    save_folder = '/home/tofi-machine/Documents/DataMining/DataMining'
    df_cleaned = load_data(file_path)
    log_datetime()

    
    

if __name__ == "__main__":
    main()