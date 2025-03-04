# Workforce Attrition Analysis & Prediction

## Project Overview

This project aims to predict employee attrition using machine learning techniques and provide actionable insights for HR strategies. The analysis utilizes the IBM HR Analytics Employee Attrition & Performance dataset from Kaggle to identify key factors contributing to employee turnover.

## Dataset

* **Source:** IBM HR Analytics Employee Attrition & Performance Dataset (Kaggle)
* **Description:** Contains various employee attributes, including demographics, job-related information, and attrition status.

## Methodology

1.  **Data Acquisition:**
    * Downloaded the dataset from Kaggle.
2.  **Data Exploration and Preprocessing:**
    * Loaded the dataset using Pandas.
    * Performed exploratory data analysis (EDA) to understand the data distribution and relationships.
    * Handle missing values (if any).
    * Converted categorical variables to numerical using one-hot encoding.
    * Scaled numerical features using StandardScaler.
    * Split the data into training and testing sets.
3.  **Model Training:**
    * Trained a Logistic Regression model using Scikit-learn.
4.  **Model Evaluation:**
    * Evaluated the model's performance using classification reports and confusion matrices.
    * Extracted feature importance to identify key predictors of attrition.
5.  **Visualization:**
    * Visualized key findings using Matplotlib and Seaborn within the notebook.
    * Created visualizations of EDA, and model results.
6.  **Actionable Insights:**
    * Derived actionable insights based on the model's results and feature importance.

## Key Findings

* Overtime work is a significant predictor of employee attrition.
* Frequent business travel also increases the likelihood of attrition.
* Specific job roles (e.g., Laboratory Technician, Sales Representative) and marital status (single) are associated with higher attrition rates.

## Model Performance

* The Logistic Regression model achieved an overall accuracy of approximately 89%.
* The model demonstrated good performance in predicting employees who would not leave (Class 0).
* However, the model's performance in predicting actual attrition (Class 1) was moderate, with a precision of 0.61 and a recall of 0.44. This is likely due to the imbalanced nature of the dataset.

## Limitations and Future Improvements

* **Imbalanced Dataset:** The dataset is imbalanced, which affects the model's ability to accurately predict attrition. Future improvements could include:
    * Applying oversampling or undersampling techniques.
    * Using different evaluation metrics (e.g., AUC-ROC, precision-recall curve).
* **Model Selection:** Exploring other machine learning models, such as Random Forest or XGBoost, may improve performance, especially in handling imbalanced data.
* **Hyperparameter Tuning:** Implementing hyperparameter tuning using GridSearchCV or RandomizedSearchCV could optimize the model's parameters.
* **Feature Engineering:** Creating new features or transforming existing ones could provide additional insights and improve model accuracy.

## Visualizations

The following visualizations are included in the Jupyter Notebook:

* Attrition Count
* Attrition by Department
* Attrition by Job Role
* Attrition by Overtime
* Top 10 Feature Importance
* Distribution of Age
* Monthly Income by Attrition (Boxplot)
