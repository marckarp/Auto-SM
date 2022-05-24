import sklearn
from sklearn import datasets #Boston Housing
import pandas as pd
import os

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target)
y.columns=['TARGET']
df = pd.concat([X,y], axis=1)

#split into train and test to push to local
bostonTrain = df.iloc[:450,:]
bostonTest = df.iloc[451:,:]

DATASET_PATH = './Data/Boston'
os.makedirs(DATASET_PATH, exist_ok=True)
bostonTrain.to_csv('Data/Boston/train.csv', index=False)
bostonTest.to_csv('Data/Boston/test.csv', index=False)