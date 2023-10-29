import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix ,precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data_url = "adult.csv"
#column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
data = pd.read_csv(data_url)

#Handle missing values
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Encode categorical variables
# categorical_columns = data.select_dtypes(include=['object']).columns
# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
data.drop('native.country', inplace=True, axis=1)
le = LabelEncoder()
data['workclass'] = le.fit_transform(data['workclass'])
data['education'] = le.fit_transform(data['education'])
data['race'] = le.fit_transform(data['race'])
data['occupation'] = le.fit_transform(data['occupation'])
data['marital.status'] = le.fit_transform(data['marital.status'])
data['relationship'] = le.fit_transform(data['relationship'])
data['sex'] = le.fit_transform(data['sex'])
data['income'] = le.fit_transform(data['income'])

print(data.columns)

# Split the data into features (X) and target (y)
X = data.drop(['race', 'relationship', 'marital.status', 'income'], axis=1)  # Features
y = data["income"]  # Target variable

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Visualize the Decision Tree (optional)
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, feature_names=list(X.columns), class_names=["<=50K", ">50K"], filled=True, rounded=True, fontsize=10)
plt.show()
