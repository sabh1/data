# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
url = "titanic.csv"
df = pd.read_csv(url)

# Step 1: Data Preparation
# Handle missing data
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
categorical_cols = ['Sex', 'Embarked']
numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Step 4: Model Building
# Split the data into features (X) and target variable (y)
X = df[categorical_cols + numeric_cols]
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Building
# Create a preprocessor for handling both categorical and numeric data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Create a pipeline with preprocessor and Gaussian Naive Bayes classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# Fit the model to the training data
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
