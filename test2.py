import pandas as pd
from ast import literal_eval  # Use literal_eval to safely evaluate strings as Python literals
from sklearn.preprocessing import MultiLabelBinarizer
import re
from collections import Counter

# Load your dataset
file_path = '/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Drop rows with any empty cells
df_cleaned = df.dropna()

# Optionally, reset the index of the cleaned DataFrame
df_cleaned.reset_index(drop=True, inplace=True)

df_cleaned.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
df_cleaned = pd.read_excel(file_path)

df = df_cleaned


def correct_text(text):
    # Define a list of common English contractions and their expanded forms
    contractions = {
        "n't": " not",
        "'m": " am",
        "'ll": " will",
        "'ve": " have",
        "'re": " are",
        "'s": "is",   # e.g., he's => he is
        "'re": "are",  # e.g., you're => you are
        "'ll": "will",  # e.g., I'll => I will
        "'ve": "have",  # e.g., they've => they have
        "'d": "had",  # e.g., he'd => he had
        "'d": "would",  # e.g., he'd => he would
        "'m": "am",  # e.g., I'm => I am
        "n't": "not"  # e.g., can't => cannot
    }

    # Replace contractions
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Correct typos using a simple word frequency-based approach
    words = re.findall(r'\w+', text.lower())
    word_counts = Counter(words)
    corrected_words = [word if word_counts[word] > 1 else max(word_counts, key=word_counts.get) for word in words]
    corrected_text = ' '.join(corrected_words)

    # Remove extra spaces and leading/trailing spaces
    corrected_text = ' '.join(corrected_text.split())
    return corrected_text


df['Genre'] = df['Genre'].apply(correct_text)


# Save the modified DataFrame back to the Excel file
#df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx', index=False)  # Replace with your file path
