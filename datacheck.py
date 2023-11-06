import pandas as pd
import numpy as np
from logger import log
from datetime import datetime


# Load the Excel file
file_path = '/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(file_path)
# Get the current date and time
current_datetime = datetime.now()

# Format and log the current date and time
formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

# ANSI escape code for red text
red_text = '\033[91m'

# ANSI escape code to reset text color
reset_text_color = '\033[0m'

# Print the formatted date and time in red
log(f"{red_text}{formatted_datetime}{reset_text_color}")


# Calculate the percentage of missing values in each column
missing_percent = (df.isnull().sum() / len(df)) * 100

# Columns with more than 2.5% missing values
columns_with_missing_values = missing_percent[missing_percent > 2.5]

# Fill NaN values in "Oscar Winners" with "not an Oscar winner"
df['Oscar Winners'] = df['Oscar Winners'].fillna("Null")

# Columns with more than 97% missing values
columns_to_delete = missing_percent[missing_percent > 2.5].index

# Delete the columns with more than 97% missing values
df.drop(columns=columns_to_delete, inplace=True)


# Print the columns with more than 2.5% missing values
if not columns_with_missing_values.empty:
    log("Columns with more than 2.5% missing values:")
    for column, percent in columns_with_missing_values.items():
        log(f"{column}: {percent:.2f}%")
else:
    log("No columns with more than 2.5% missing values.")

# Check for duplicates in the "Film" column
duplicates = df[df.duplicated(subset="Film", keep=False)]

# Print the duplicate rows
if not duplicates.empty:
    log("Duplicate rows in the 'Film' column:")
    log(str(duplicates))
else:
    log("No duplicates found in the 'Film' column.")

# Drop the duplicate rows, keeping the first occurrence
df = df.drop(duplicates.index)

log("Duplicate rows in the 'Film' column have been dropped, and the DataFrame has been saved to the Excel file.")


# Save the modified DataFrame back to the Excel file
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
log("Excel file updated")