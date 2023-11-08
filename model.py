import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset into a DataFrame
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx")

# Data Cleaning and Feature Engineering - You need to perform these steps

# Create the target variable
# Assume you have a column "Oscar Winner" with 1 for Oscar winners and 0 for others
y = df["Oscar Winners"]
# Use the exact column names in your code
X = df[["Year", "Script Type", "Rotten Tomatoes  critics", "Metacritic  critics", "Average critics", "Rotten Tomatoes Audience", "Metacritic Audience", "Average audience", "Audience vs Critics deviance", "Genre", "Domestic Gross", "Domestic gross ($million)", "Foreign Gross ($million)", "Foreign Gross", "Worldwide Gross", "Worldwide Gross ($million)", "Release Date (US)"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print(report)
