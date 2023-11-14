import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the pre-trained model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.load_model("./trained_model.pkl")

# Load the new dataset
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/moviesNEW.xlsx")

# Data Cleaning and Feature Engineering - You need to perform these steps
df['Domestic gross ($million)'] = df['Domestic gross ($million)'].replace(',', '', regex=True).astype(float)
df['Domestic Gross'] = df['Domestic Gross'].replace(',', '', regex=True).astype(float) 
df['Foreign Gross ($million)'] = df['Foreign Gross ($million)'].replace(',', '', regex=True).astype(float)
df['Foreign Gross'] = df['Foreign Gross'].replace(',', '', regex=True).astype(float)           
df['Worldwide Gross'] = df['Worldwide Gross'].replace(',', '', regex=True).astype(float)   

# Apply the same transformations that were applied to the training data
X_new = new_data.drop('one-hot encoding Oscar Winners', axis=1)

# Make predictions on the new dataset
y_pred_new = rf_classifier.predict(X_new)

# Display the predictions
print("Predictions:")
print(y_pred_new)
