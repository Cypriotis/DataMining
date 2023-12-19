import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset into a DataFrame
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx")

# Step 1: Separate features (X) and target variable (y)
X = df.drop('one-hot encoding Oscar Winners', axis=1)
y = df['one-hot encoding Oscar Winners']

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Create and train a Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train_resampled, y_train_resampled)

# Save the trained model to a file
joblib.dump(gb_classifier, "/home/tofi-machine/Documents/DataMining/DataMining/trained_model_gb.pkl")

# Step 5: Perform 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(gb_classifier, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')

# Display cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Step 6: Make predictions on the test set
y_pred = gb_classifier.predict(X_test)

# Step 7: Evaluate the Gradient Boosting model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the evaluation metrics
print(f'Gradient Boosting Model Accuracy: {accuracy}')
print('Gradient Boosting Classification Report:')
print(classification_rep)
