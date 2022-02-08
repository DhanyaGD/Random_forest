"""Random Forest

Assignment


About the data:
Let’s consider a Company dataset with around 10 variables and 400 records.
The attributes are as follows:
 Sales -- Unit sales (in thousands) at each location
 Competitor Price -- Price charged by competitor at each location
 Income -- Community income level (in thousands of dollars)
 Advertising -- Local advertising budget for company at each location (in thousands of dollars)
 Population -- Population size in region (in thousands)
 Price -- Price company charges for car seats at each site
 Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
 Age -- Average age of the local population
 Education -- Education level at each location
 Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
 US -- A factor with levels No and Yes to indicate whether the store is in the US or not
The company dataset looks like this:

Problem Statement:
A cloth manufacturing company is interested to know about the segment or attributes causes high sale.
Approach - A Random Forest can be built with target variable Sales (we will first convert it in categorical variable) & all other variable will be independent in the analysis.
"""


import numpy as np
import pandas as pd

company = pd.read_csv('C:/ExcelrData/Data-Science_Assignments/Random_forest/Company_Data.csv')
company.columns
company.isna().sum()
company.Sales.median()

# create bins for sales
cut_labels = ['low', 'medium', 'high']
cut_bins = [-1, 5.66, 12, 17]
company['sales'] = pd.cut(company['Sales'], bins=cut_bins, labels=cut_labels)
company.pop('Sales')

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
company['ShelveLoc'] = label_encoder.fit_transform(company['ShelveLoc'])
company['Urban'] = label_encoder.fit_transform(company['Urban'])
company['US'] = label_encoder.fit_transform(company['US'])

array = company.values
X = array[:, 0:10]
Y = array[:, 10]
# splitting data using K-Fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7, shuffle=True)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_features=3)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
