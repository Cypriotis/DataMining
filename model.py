import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Assuming df is your DataFrame and 'one-hot encoding Oscar Winners' is the column indicating Oscar win or not
# Load the dataset into a DataFrame
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx")

# Data Cleaning and Feature Engineering - You need to perform these steps
df['Domestic gross ($million)'] = df['Domestic gross ($million)'].replace(',', '', regex=True).astype(float)
df['Domestic Gross'] = df['Domestic Gross'].replace(',', '', regex=True).astype(float) 
df['Foreign Gross ($million)'] = df['Foreign Gross ($million)'].replace(',', '', regex=True).astype(float)
df['Foreign Gross'] = df['Foreign Gross'].replace(',', '', regex=True).astype(float)           
df['Worldwide Gross'] = df['Worldwide Gross'].replace(',', '', regex=True).astype(float)        

# Step 1: Separate features (X) and target variable (y)
X = df.drop('one-hot encoding Oscar Winners', axis=1)
y = df['one-hot encoding Oscar Winners']

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Create and train a random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_resampled, y_train_resampled)

# Save the trained model to a file
joblib.dump(rf_classifier, "./trained_model.pkl")

# Step 5: Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the evaluation metrics
print(f'Model Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)
