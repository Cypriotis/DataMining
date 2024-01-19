import pandas as pd

# File paths - Replace these with the actual paths in your development environment
movies_metadata_path = 'movies_metadata.csv'
movies_path = 'movies.xlsx'

def find_matching_films(movies_path, movies_metadata_path):
    # Load the data
    movies_metadata = pd.read_csv(movies_metadata_path, encoding='ISO-8859-1', low_memory=False)
    movies = pd.read_excel(movies_path)

    # Convert the 'title' column in movies_metadata to string to ensure consistency
    movies_metadata['title'] = movies_metadata['title'].astype(str).str.strip().str.lower()

    # Make sure the Film column in movies is in a consistent format
    movies['Film'] = movies['Film'].astype(str).str.strip().str.lower()

    # Initialize a new column for vote_average in movies
    movies['vote_average'] = 5  # Default value

    # Iterate over each film in movies
    for index, row in movies.iterrows():
        # Try to find the film in movies_metadata
        matching_row = movies_metadata[movies_metadata['title'] == row['Film']]
        if not matching_row.empty:
            # If found, update the vote_average
            movies.at[index, 'vote_average'] = matching_row['vote_average'].iloc[0]

    # Save the updated dataframe to a new CSV file
    movies.to_csv('movies_with_vote_average.csv', index=False)
    print(f"Films with their vote average have been saved to 'movies_with_vote_average.csv'")

# Call the function with the file paths
find_matching_films(movies_path, movies_metadata_path)