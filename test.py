import pandas as pd

def load_excel(file_path):
    """
    Load an Excel file into a DataFrame.

    Parameters:
    - file_path (str): Path to the Excel file.

    Returns:
    - df (pd.DataFrame): Loaded DataFrame.
    """
    try:
        # Load the data into a DataFrame
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error loading the Excel file: {e}")
        return None

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

def save_to_excel(df, file_path):
    """
    Save the DataFrame to an Excel file.

    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - file_path (str): Path to save the Excel file.
    """
    try:
        df.to_excel(file_path, index=False)
        print(f"DataFrame saved to '{file_path}'.")
    except Exception as e:
        print(f"Error saving the DataFrame to Excel: {e}")

def main():
    # Replace 'your_file_path.xlsx' with the actual path to your Excel file
    file_path = '/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx'

    columns_to_convert = [
        'Domestic gross ($million)',
        'Foreign Gross ($million)',
        'Worldwide Gross ($million)',
        'Budget ($million)'
    ]

    # Load the Excel file
    df = load_excel(file_path)

    # Check if the DataFrame is successfully loaded
    if df is not None:
        # Display basic information about the DataFrame
        print("DataFrame information:")
        print(df.info())

        # Display the first few rows of the DataFrame
        print("\nFirst 5 rows of the DataFrame:")
        print(df.head())

        # Convert the specified columns to numeric
        df_numeric = convert_columns_to_numeric(df, columns_to_convert)

        # Display the updated DataFrame
        if df_numeric is not None:
            print("\nDataFrame after converting specified columns to numeric:")
            print(df_numeric.head())

            # Save the updated DataFrame to the same Excel file
            save_to_excel(df_numeric, file_path)

if __name__ == "__main__":
    main()


    
