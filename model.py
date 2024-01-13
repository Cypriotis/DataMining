import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline  # Use imblearn's Pipeline instead of sklearn's
import joblib

# Load the dataset into a DataFrame
df = pd.read_excel("/home/tofi-machine/Documents/DataMining/DataMining/movies.xlsx")

# Data Cleaning and Feature Engineering
df['Domestic gross ($million)'] = df['Domestic gross ($million)'].replace(',', '', regex=True).astype(float)
df['Domestic Gross'] = df['Domestic Gross'].replace(',', '', regex=True).astype(float)
df['Foreign Gross ($million)'] = df['Foreign Gross ($million)'].replace(',', '', regex=True).astype(float)
df['Foreign Gross'] = df['Foreign Gross'].replace(',', '', regex=True).astype(float)
df['Worldwide Gross'] = df['Worldwide Gross'].replace(',', '', regex=True).astype(float)

# Separate features (X) and target variable (y)
X = df.drop('one-hot encoding Oscar Winners', axis=1)
y = df['one-hot encoding Oscar Winners']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with SMOTE, StandardScaler, and RandomForestClassifier
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42, class_weight="balanced"))
])

# Define hyperparameter grid for RandomForestClassifier
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Save the best-trained model to a file
best_rf_classifier = grid_search.best_estimator_
joblib.dump(best_rf_classifier, "/home/tofi-machine/Documents/DataMining/DataMining/best_trained_model.pkl")

# Make predictions on the test set
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Display the evaluation metrics
print(f'Model Accuracy: {accuracy}')
print('Classification Report:')
print(classification_rep)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

# Evaluate with Gradient Boosting
gb_classifier = GradientBoostingClassifier(random_state=42)
gb_classifier.fit(X_train, y_train)
y_pred_gb = gb_classifier.predict(X_test)

accuracy_gb = accuracy_score(y_test, y_pred_gb)
classification_rep_gb = classification_report(y_test, y_pred_gb)

# Display the evaluation metrics for Gradient Boosting
print(f'Gradient Boosting Accuracy: {accuracy_gb}')
print('Gradient Boosting Classification Report:')
print(classification_rep_gb)
