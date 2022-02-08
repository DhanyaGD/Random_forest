"""Use Random Forest to prepare a model on fraud data
treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
"""

import numpy as np
import pandas as pd

fraud = pd.read_csv('C:/ExcelrData/Data-Science_Assignments/Random_forest/Fraud_check.csv')
fraud.columns
fraud.columns = ['under_grad', 'marital_status', 'taxable_income', 'city_pop', 'work_exp', 'urban']

# creating bins for taxable_income=>to categorical
cut_labels = ['Risky', 'Good']
cut_bins = [0, 30000, 99620]
fraud['tax_inc'] = pd.cut(fraud['taxable_income'], bins=cut_bins, labels=cut_labels)
fraud.pop('taxable_income')

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
fraud['under_grad'] = label_encoder.fit_transform(fraud['under_grad'])
fraud['marital_status'] = label_encoder.fit_transform(fraud['marital_status'])
fraud['urban'] = label_encoder.fit_transform(fraud['urban'])

array = fraud.values
X = array[:, 0:5]
Y = array[:, 5]
# splitting data using K-Fold cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=10, random_state=7, shuffle=True)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_features=3)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
