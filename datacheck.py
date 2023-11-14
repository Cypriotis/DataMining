import pandas as pd
import numpy as np
from logger import log
from datetime import datetime
import plotly.express as px
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import re 
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from spellchecker import SpellChecker





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

#Essential first checks

# Fill NaN values in "Oscar Winners" with "not an Oscar winner"
df['Oscar Winners'] = df['Oscar Winners'].fillna("not an Oscar winner")

# Check for duplicates in the "Film" column
duplicates = df[df.duplicated(subset="Film", keep=False)]

# Deleting irelevant columns for our analysis
columns_to_delete = ['Film', 'Rotten Tomatoes vs Metacritic  deviance','Opening Weekend','Opening weekend ($million)',' Budget recovered',' Budget recovered opening weekend']  # Specify the columns to delete
df = df.drop(columns=columns_to_delete)

#########################
#           PART1       #
#########################

# Define the phrase you want to find and standardize
target_phrase = 'oscar winner'  # Replace with the phrase you want to find

# Define the desired formatting (e.g., 'uppercase', 'lowercase', 'titlecase')
desired_formatting = 'titlecase'  # Replace with your desired formatting

# Function to standardize the text
def standardize_text(text):
    if desired_formatting == 'uppercase':
        return text.group(0).upper()
    elif desired_formatting == 'lowercase':
        return text.group(0).lower()
    elif desired_formatting == 'titlecase':
        return text.group(0).title()
    else:
        return text.group(0)  # No formatting specified

# Convert the column to strings
df['Oscar Winners'] = df['Oscar Winners'].astype(str)

# Apply the standardization function to the column containing the target phrase
df['Oscar Winners'] = df['Oscar Winners'].apply(lambda x: re.sub(target_phrase, standardize_text, x, flags=re.IGNORECASE))


########################
#        PART 2        #
########################

# Calculate the percentage of missing values in each column
missing_percent = (df.isnull().sum() / len(df)) * 100

# Columns with more than 2.5% missing values
columns_with_missing_values = missing_percent[missing_percent > 2.5]

# Columns with more than 97% missing values
columns_to_delete = missing_percent[missing_percent > 2.5].index

# Delete the columns with more than 97% missing values
df.drop(columns=columns_to_delete, inplace=True)

# Drop rows with any empty cells
df_cleaned = df.dropna()

# Optionally, reset the index of the cleaned DataFrame
df_cleaned.reset_index(drop=True, inplace=True)

df_cleaned.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
df_cleaned = pd.read_excel(file_path)

df = df_cleaned



# Print the columns with more than 2.5% missing values
if not columns_with_missing_values.empty:
    log("Columns with more than 2.5% missing values:")
    for column, percent in columns_with_missing_values.items():
        log(f"{column}: {percent:.2f}%")
else:
    log("No columns with more than 2.5% missing values.")


# Print the duplicate rows
if not duplicates.empty:
    log("Duplicate rows in the 'Film' column:")
    log(str(duplicates))
else:
    log("No duplicates found in the 'Film' column.")

# Drop the duplicate rows, keeping the first occurrence
df = df.drop(duplicates.index)

log("Duplicate rows in the 'Film' column have been dropped, and the DataFrame has been saved to the Excel file.")

#######################
#        PART 3       #
#######################


# Get the list of numeric columns
numeric_columns = df.select_dtypes(include='number').columns

# Initialize a list to store column names with min value 0
columns_with_min_zero = []

# Describe and plot each numeric column
for column in numeric_columns:
    description = df[column].describe()

    # Check if the minimum value is 0
    if description['min'] == 0:
        columns_with_min_zero.append(column)
    
# Delete rows where the minimum value is 0 in specified columns
for column in columns_with_min_zero:
    df = df[df[column] != 0]

###########################
#        PART 4           #
###########################

# Function to check if any cell in a row contains the "-" symbol
def contains_minus_without_number(row):
    return any('-' in str(cell) and not any(char.isdigit() for char in str(cell).split('-')[1]) for cell in row)

# Mark rows for deletion based on the condition
rows_to_delete = df.apply(contains_minus_without_number, axis=1)

# Delete the marked rows from the DataFrame
df = df[~rows_to_delete]

# Drop rows with any empty cells
df_clean = df.dropna()

############################
#         PART 5           #
############################
# One hot encoding
enc = preprocessing.OrdinalEncoder()
enc.fit(df[["Script Type"]]) 
df['one-hot encoding ScriptType']=enc.transform(df[["Script Type"]])

# Save the modified DataFrame back to the Excel file
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
df = pd.read_excel(file_path)

# Specify the column containing lists of strings
column_name = 'Genre'

# Initialize the SpellChecker
spell = SpellChecker()

# Function to correct spelling in a given cell
def correct_spelling(cell_content, spell_checker):
    if pd.notna(cell_content):
        words = [word.strip() for word in cell_content.split(',')]
        corrected_words = [spell_checker.correction(word) if spell_checker.correction(word) else word for word in words]
        return ', '.join(corrected_words)
    else:
        return cell_content
    
# Apply the correction function to the specified column
df[column_name] = df[column_name].apply(lambda x: correct_spelling(x, spell))

# Save the modified DataFrame back to the Excel file
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
df = pd.read_excel(file_path)


# Specify the column containing strings with commas
column_name = 'Genre'

# Function to remove anything after a comma
def remove_after_comma(cell_content):
    if pd.notna(cell_content):
        return cell_content.split(',')[0]  # Keep only the content before the first comma
    else:
        return cell_content

# Apply the function to the specified column
df[column_name] = df[column_name].apply(remove_after_comma)
# Save the modified DataFrame back to the Excel file
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
df = pd.read_excel(file_path)

# One hot encoding
enc = preprocessing.OrdinalEncoder()
enc.fit(df[["Genre"]]) 
df['one-hot encoding Genre']=enc.transform(df[["Genre"]])

enc = preprocessing.OrdinalEncoder()
enc.fit(df[["Oscar Winners"]]) 
df['one-hot encoding Oscar Winners']=enc.transform(df[["Oscar Winners"]])

# Specify the column you want to invert
column_name_to_invert = 'one-hot encoding Oscar Winners'

# Invert the values in the specified column
df[column_name_to_invert] = df[column_name_to_invert].apply(lambda x: 1 if x == 0 else 0)

# Specify the target column and the deletion probability
target_column = 'column_name_to_invert'  # Replace with the actual column name
deletion_probability = 0.9  # 50% chance

# Create a mask based on the condition (cell value equals 0)
delete_mask = (df[column_name_to_invert] == 0)

# Apply random deletion based on the probability
df = df[~(delete_mask & (np.random.rand(len(df)) < deletion_probability))]

df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
df = pd.read_excel(file_path)

# Display basic information about the dataset
print("Number of rows and columns:", df.shape)
print("Column names:", df.columns)
print("Data types of columns:")
print(df.dtypes)

# Count the total number of records
total_records = len(df)

# Count the number of records that won an Oscar
oscar_winners = df[df['Oscar Winners'] == 'Oscar Winner']
num_oscar_winners = len(oscar_winners)

# Calculate the percentage of records that won an Oscar
percentage_oscar_winners = (num_oscar_winners / total_records) * 100

# Calculate the percentage of records that didn't win an Oscar
percentage_non_oscar_winners = 100 - percentage_oscar_winners

# Print the results
print(f"Total records: {total_records}")
print(f"Number of Oscar winners: {num_oscar_winners}")
print(f"Percentage of Oscar winners: {percentage_oscar_winners:.2f}%")
print(f"Percentage of non-Oscar winners: {percentage_non_oscar_winners:.2f}%")

# Create a Plotly pie chart
fig = px.pie(
    names=['Oscar Winners', 'notyet'],
    values=[percentage_oscar_winners, percentage_non_oscar_winners],
    title=f"Percentage of Oscar Winners in the Dataset ({total_records} records)"
)

# Specify the folder and filename for saving the PNG image
save_folder = '/home/tofi-machine/Documents/DataMining/DataMining'  # Replace with the folder path where you want to save the image
save_filename = 'oscar_winners_pie_chart.png'

# Create the full path to save the image
save_path = os.path.join(save_folder, save_filename)

# Save the figure as a PNG image
fig.write_image(save_path, format='png')

# Correlation heatmap for numeric columns
numeric_cols = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for selected numeric columns
sns.pairplot(df[['Budget ($million)', 'Domestic gross ($million)', 'Foreign Gross ($million)', 'Worldwide Gross ($million)']])
plt.title("Pairplot of Selected Numeric Columns")
plt.show()

# Countplot for Oscar Winners
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Oscar Winners')
plt.title("Count of Oscar Winners")
plt.show()

for column in numeric_columns:
    # Describe the numeric column
    description = df[column].describe()
    print(f"Descriptive Statistics for {column}:")
    print(description)

    # Plot a histogram for the numeric column
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Print the updated DataFrame without rows where the minimum value is 0
print("Updated DataFrame after deleting rows:")
print(df)

# Save the modified DataFrame back to the Excel file
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
log("Excel file updated")