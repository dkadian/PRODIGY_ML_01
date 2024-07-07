# Prodigy Infotech Internship task 1 of Machine Learning
## Linear Regression model to predict prices of houses

### Overview
The code aims to build and evaluate linear regression models to predict house prices (SalePrice) based on their square footage and number of bedrooms and bathrooms, using a dataset containing information about residential properties. Two variations of linear regression models are demonstrated: one with polynomial features and another without.

### Dataset

The dataset consists of:
**1. train.csv** - the training set
**2. test.csv** - the test set
**3. data_description.txt** - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
**4. sample_submission.csv** - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms

**Dataset link:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=test.csv

### Technologies Used:

**1.Python:** The primary programming language for the entire solution.

**2.NumPy:** Library used for numerical operations and data handling.

**3.Matplotlib:** Library for data and image visualization, here used for visulaing results.

**4.scikit-learn:** Provides machine learning tools for building regression models, feature scaling, and polynomial feature generation.

**5.pandas:** Utilized for data manipulation and analysis.

### Code Overview:

**1.Dataset Loading and Cleaning:** The dataset is loaded from CSV files (train.csv and test.csv) and missing values in specific columns are dropped. Outliers in GrLivArea and SalePrice columns are removed for illustration purposes.

**2.Feature Engineering:** A new feature, TotalBathrooms, is created by combining the number of full and half bathrooms.

**3.Selecting Features and Target Variable:** Relevant features (GrLivArea, BedroomAbvGr, FullBath, and TotalBathrooms) and the target variable (SalePrice) are selected.

**4.Data Splitting:** The dataset is split into training and testing sets.

**5.Feature Scaling:** Standard scaling is applied to the features to ensure they have similar scales.

**6.Polynomial Features:** Polynomial features up to the second degree are generated from the scaled features.

**7.Linear Regression Model Training:** A linear regression model is trained using the polynomial features.

**8.Model Evaluation:** The model is evaluated on the test set using metrics such as mean squared error, mean absolute error, R-squared score, and root mean squared error.

**9.Visualization:** The predictions and actual prices for the living area (GrLivArea) are visualized using a scatter plot.


### Conclusion:
The code demonstrates the application of linear regression models for predicting house prices based on selected features. The R-squared score of **0.75** suggests that **75%** of the variance in house prices is explained by the model. The inclusion of polynomial features allows the model to capture non-linear relationships in the data. The evaluation metrics and visualizations provide insights into the performance of the models on the test data.
