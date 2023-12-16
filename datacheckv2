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

from datacheck import standardize_text

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

# Function to fill missing values in the 'Oscar Winners' column
def fill_missing_oscar_values(df):
    # Print the columns before filling missing values
    print("Columns before filling missing values:", df.columns)

    # If 'one-hot encoding Oscar Winners' column is present, no need to fill NaN values
    # df['Oscar Winners'] = df['Oscar Winners'].fillna("not an Oscar winner")

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
def standardize_text_column(df, target_phrase, desired_formatting):
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

def one_hot_encoding(df, column_name):
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' removes one of the one-hot encoded columns to avoid multicollinearity

    # Fit and transform the specified column
    encoded_values = encoder.fit_transform(df[[column_name]])

    # Create new column names for the one-hot encoded columns
    new_columns = [f'{column_name}_{category}' for category in encoder.get_feature_names_out([column_name])]

    # Create a DataFrame with the one-hot encoded columns
    df_encoded = pd.DataFrame(encoded_values, columns=new_columns)

    # Concatenate the original DataFrame and the one-hot encoded DataFrame
    df = pd.concat([df, df_encoded], axis=1)

    # Drop the original column
    df = df.drop(columns=[column_name])

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
    return df

# Function to invert the values in a specified column
def invert_column_values(df, column_name_to_invert):
    df[column_name_to_invert] = df[column_name_to_invert].apply(lambda x: 1 if x == 0 else 0)
    return df

# Function to apply random deletion based on a deletion probability to a specified column
def random_deletion(df, column_name_to_invert, deletion_probability):
    delete_mask = (df[column_name_to_invert] == 0)
    df = df[~(delete_mask & (np.random.rand(len(df)) < deletion_probability))]
    return df

# Function to drop irrelevant columns from PART3
def drop_irrelevant_columns(df, columns_to_delete):
    if all(col in df.columns for col in columns_to_delete):
        df = df.drop(columns=columns_to_delete, axis=1)
    return df

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
    df = load_data(file_path)

    log_datetime()
    fill_missing_oscar_values(df)
    drop_duplicates(df)

    columns_to_delete_part1 = ['Film', 'Rotten Tomatoes vs Metacritic  deviance', 'Opening Weekend',
                                'Opening weekend ($million)', ' Budget recovered', ' Budget recovered opening weekend']
    drop_irrelevant_columns(df, columns_to_delete_part1)

    target_phrase = 'oscar winner'
    desired_formatting = 'titlecase'
    standardize_text_column(df, target_phrase, desired_formatting)

    columns_to_delete_part2 = calculate_missing_percent(df)[calculate_missing_percent(df) > 2.5].index
    delete_columns_with_missing_values(df, columns_to_delete_part2)

    df_cleaned = drop_rows_with_empty_cells(df)

    save_data(df_cleaned, file_path)
    df_cleaned = load_data(file_path)

    print("Columns with more than 2.5% missing values:")
    print(calculate_missing_percent(df_cleaned)[calculate_missing_percent(df_cleaned) > 2.5])

    numeric_columns = df_cleaned.select_dtypes(include='number').columns
    check_min_value_zero(df_cleaned, numeric_columns)

    df_clean = delete_rows_with_minus_symbol(df_cleaned)

    column_name_to_encode = 'Script Type'
    one_hot_encoding(df_clean, column_name_to_encode)

    apply_spell_checker(df_clean, 'Genre')
    remove_text_after_comma(df_clean, 'Genre')
    one_hot_encoding(df_clean, 'Genre')
    one_hot_encoding(df_clean, 'Oscar Winners')

    column_name_to_invert = 'one-hot encoding Oscar Winners'
    invert_column_values(df_clean, column_name_to_invert)
    random_deletion(df_clean, column_name_to_invert, 0.90)

    columns_to_delete_part3 = ['Genre', 'Script Type', 'Release Date (US)', ' of Gross earned abroad']
    drop_irrelevant_columns(df_clean, columns_to_delete_part3)

    save_data(df_clean, file_path)
    df_clean = load_data(file_path)

    display_basic_info(df_clean)
    analyze_oscar_winners(df_clean)
    save_pie_chart(df_clean, save_folder, len(df_clean))
    plot_correlation_heatmap(df_clean)
    plot_pairplot(df_clean)
    plot_countplot(df_clean)
    plot_histogram(df_clean, numeric_columns)

    print("Updated DataFrame after deleting rows:")
    print(df_clean)

    save_data(df_clean, file_path)
    print("Excel file updated")
