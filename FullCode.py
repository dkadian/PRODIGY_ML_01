# House Price Prediction Using Linear Regression Model
### Import All the necessary Libraries
#%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, QuantileTransformer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.kernel_ridge import KernelRidge


### Working on Train Dataframe
train_df = pd.read_csv('train.csv')
train_df.columns
numeric_df = train_df.select_dtypes(include='number')
correlation_matrix = numeric_df.corr()
correlation_matrix['SalePrice'].sort_values(ascending=False)
req_tr = ["GarageArea", "OverallQual", "TotalBsmtSF", "1stFlrSF","2ndFlrSF", "LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","TotRmsAbvGrd","SalePrice"]
selected_tr = train_df[req_tr]
selected_tr.loc[:, 'TotalBath'] = (selected_tr['BsmtFullBath'].fillna(0)+
                                   selected_tr['BsmtHalfBath'].fillna(0)+
                                   selected_tr['FullBath'].fillna(0)+
                                   selected_tr['HalfBath'].fillna(0))

selected_tr.loc[:, 'TotalSF'] = (selected_tr['TotalBsmtSF'].fillna(0)+
                                   selected_tr['1stFlrSF'].fillna(0)+
                                   selected_tr['2ndFlrSF'].fillna(0)+
                                   selected_tr['LowQualFinSF'].fillna(0)+
                                   selected_tr['GrLivArea'].fillna(0))
selected_tr
### Keeping Only the Necessary Columns
train_df = selected_tr[['TotRmsAbvGrd','TotalBath','GarageArea','TotalSF','OverallQual','SalePrice']]

train_df
### Splitting the dataset and Creating pipeline
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(train_df, test_size=0.2,random_state=42)
print(f"Rows in train Set : {len(train_set)} \nRows in test Set : {len(test_set)}")
housing = train_set.drop("SalePrice", axis = 1)
housing_lables = train_set["SalePrice"].copy()
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler())
])
x_train = my_pipeline.fit_transform(housing)

x_train
Y_train = housing_lables
Y_train.shape
### Correlations
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
#%matplotlib inline
sns.pairplot(train_df)
plt.tight_layout()
plt.show

corr_martix = train_df.corr()
corr_martix['SalePrice'].sort_values(ascending=False)

sns.heatmap(train_df.corr(), annot=True)
### Working with test Dataframe
test_df = pd.read_csv("test.csv")
test_df.head()
req_tst = ["GarageArea", "OverallQual", "TotalBsmtSF", "1stFlrSF","2ndFlrSF", "LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath","FullBath","HalfBath","TotRmsAbvGrd"]

selected_tst = test_df[req_tst]
selected_tst.loc[:, 'TotalBath'] = (selected_tst['BsmtFullBath'].fillna(0)+
                                   selected_tst['BsmtHalfBath'].fillna(0)+
                                   selected_tst['FullBath'].fillna(0)+
                                   selected_tst['HalfBath'].fillna(0))

selected_tst.loc[:, 'TotalSF'] = (selected_tst['TotalBsmtSF'].fillna(0)+
                                   selected_tst['1stFlrSF'].fillna(0)+
                                   selected_tst['2ndFlrSF'].fillna(0)+
                                   selected_tst['LowQualFinSF'].fillna(0)+
                                   selected_tst['GrLivArea'].fillna(0))
selected_tst
test_df_unproc = selected_tst[['TotRmsAbvGrd','TotalBath','GarageArea','TotalSF','OverallQual']]
test_df_unproc
test_df = test_df_unproc.fillna(test_df_unproc.mean())

x_test = my_pipeline.transform(test_df[['TotRmsAbvGrd','TotalBath','GarageArea','TotalSF','OverallQual']].values)
x_test
### Model Selection
#model = LinearRegression()

#model = DecisionTreeREgressor()
model = RandomForestRegressor()
model.fit(x_train,Y_train)
y_train_pred = model.predict(x_train)
y_train_pred[:5]
some_data = housing.iloc[:5]
some_labels = housing_lables.iloc[:5]

proc_data = my_pipeline.transform(some_data)

model.predict(proc_data)
list(some_labels)
train_mse = mean_squared_error(Y_train,y_train_pred)
train_rmse = np.sqrt(train_mse)
print(f"Training MSE : {train_mse: .2f}, Training RMSE : {train_rmse: .2f}")
### Cross-Validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,x_train,Y_train,scoring="neg_mean_squared_error",cv = 200)
rmse_scores = np.sqrt(-scores)
rmse_scores
def print_scores(scores):
    print("Scores : ",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation: ",scores.std())
print_scores(rmse_scores)
y_pred = model.predict(x_test)
y_pred
pred = pd.DataFrame(y_pred)
sub_df = pd.read_csv('sample_submission.csv')
datasets = pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns = ['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)