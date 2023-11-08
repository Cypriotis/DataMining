import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


# Load the Excel file
file_path = '/home/tofi-machine/Documents/DataMining/DataMining/moviess.xlsx'  # Replace with the path to your Excel file
df = pd.read_excel(file_path)

# Initialize an empty list to store all the genres
all_genres = []

# Iterate through the 'Genre' column and split the genres
for genre_string in df['Genre']:
    if isinstance(genre_string, str):
        genres = [genre.strip() for genre in genre_string.split(',')]
        all_genres.extend(genres)

# Convert the unique genres to title case
title_case_genres = [genre.title() for genre in all_genres]


# Remove duplicate genres by converting the list to a set and back to a list
unique_genres = list(set(title_case_genres))

print(title_case_genres)

# Convert the list to a set to remove duplicates
unique_set = set(title_case_genres)

print("test")
# Iterate through the set and print each unique element
for element in unique_set:
    print(element)

enc = preprocessing.OrdinalEncoder()
enc.fit(unique_set) 
df['genres']=enc.transform(unique_set)




# Save the modified DataFrame back to the Excel file
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/moviess.xlsx', index=False)  # Replace with your file path
