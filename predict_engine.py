import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the pre-trained model
rf_classifier = joblib.load("/home/tofi-machine/Documents/DataMining/DataMining/trained_model.pkl")

# Load the new dataset
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx")


# Apply the same transformations that were applied to the training data
X_new = df.drop('one-hot encoding Oscar Winners', axis=1)

# Make predictions on the new dataset
y_pred_new = rf_classifier.predict(X_new)

# Display the predictions
print("Predictions:")
print(y_pred_new)

X_new.to_excel('/home/tofi-machine/Documents/DataMining/DataMining/test.xlsx', index=False)  # Replace with your file path