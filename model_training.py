import pandas as pd

# Load the cleaned dataset
df_cleaned = pd.read_csv("OnlineRetail_cleaned.csv")

# Step 1: Select relevant features
df_model = df_cleaned[['CustomerID', 'Quantity', 'UnitPrice']]

# Drop rows with missing values
df_model = df_model.dropna()

print(df_model.head())  # Verify feature selection

#Step 1: Import Required Libraries
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Step 2: Prepare the Data
# Create a new column to categorize customers into 'High' and 'Low' value groups based on Quantity
df_model['CustomerGroup'] = np.where(df_model['Quantity'] > df_model['Quantity'].median(), 'High', 'Low')

# Split data into features (X) and target (y)
X = df_model[['Quantity', 'UnitPrice']]  # Features: Quantity and UnitPrice
y = df_model['CustomerGroup']  # Target: High-value or Low-value customers

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("✅ Data prepared for training!")



#Step 3: Train a Random Forest Classifie
# Train a Random Forest Classifier
# Evaluate the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

#Step 4: Evaluate the Model
# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Step 5: Save the Model
import joblib

# Save the Random Forest model
joblib.dump(rf_classifier, "customer_segmentation_model.pkl")
print("✅ Model saved successfully!")


import matplotlib.pyplot as plt

# Plot feature importance
feature_importance = rf_classifier.feature_importances_
plt.bar(X.columns, feature_importance, color='skyblue')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()


import seaborn as sns

# Plot Customer Group Distribution
sns.countplot(x=y, palette='viridis')
plt.title("Customer Group Distribution")
plt.xlabel("Customer Group")
plt.ylabel("Count")
plt.show()