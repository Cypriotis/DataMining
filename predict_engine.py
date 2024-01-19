import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load the pre-trained model
rf_classifier = joblib.load("/home/tofi-machine/Documents/DataMining/DataMining/best_trained_model.pkl")

# Load the new dataset
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/main.xlsx")

# Apply the same transformations that were applied to the training data
X_new = df.drop('one-hot encoding Oscar Winners', axis=1)

# Make predictions on the new dataset
y_pred_new = rf_classifier.predict(X_new)

# Create a DataFrame for predictions
predictions_df = pd.DataFrame({'Predictions': y_pred_new})

# Display and save the predictions to an Excel file with a single column
print("Predictions:")
print(predictions_df)

predictions_df.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/results_it22131.xlsx', index=False)

# Count the occurrences of '1' in the 'Predictions' column
count_ones = predictions_df['Predictions'].sum()
print(f"Number of 1's in the 'Predictions' column: {count_ones}")
