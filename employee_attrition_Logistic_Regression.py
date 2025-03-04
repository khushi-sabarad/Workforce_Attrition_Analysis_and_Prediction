# dataset from kaggle: IBM HR Analytics Employee Attrition & Performance
# Faker: python library which we can use to create a synthetic dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("C:/Users/Khushi/OneDrive/Desktop/coding/Workforce_Attrition_Analysis_and_Prediction/employee_attrition.csv")

# Basic data exploration
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (if any)
# df = df.dropna() or df['column'].fillna(df['column'].mean(), inplace=True)

# Data visualization (EDA)
sns.countplot(x='Attrition', data=df)
plt.show()

# Explore relationships between variables
sns.countplot(x='Department', hue='Attrition', data=df)
plt.show()

#Convert the target variable (Attrition) to numerical.
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

# One hot encode categorical variables to numerical (conversion)
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature selection (dropping irrelevant columns)
df = df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1)

#Scale numerical data.
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
numerical_cols.remove('Attrition') #remove the target variable from being scaled
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#Split data into training and testing sets.
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train a logistic regression model
model = LogisticRegression(max_iter=1000) #increase max_iter if needed.
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': abs(model.coef_[0])})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance.head(10))