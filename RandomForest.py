from sklearn import ensemble 
from sklearn.cross_validation import train_test_split
import pandas as pd 
import numpy as np 


def strToFloat(data):
	stringNums = {}
	i = 0
	for d in data:
		if d not in stringNums:
			stringNums[d] = i
			i += 1

	return [stringNums[d] for d in data]

df = pd.read_csv("data.csv")

train, test = train_test_split(df, test_size = 0.2)

print train.shape
print test.shape

attribute_cols = [col for col in train.columns if col not in ["GN/LS"]]

X_train = train[attribute_cols]
X_test = test[attribute_cols]

X_train = list(map(strToFloat, X_train.columns))
X_test = list(map(strToFloat, X_test.columns))

y = train["GN/LS"]
print(y)
y = list(map(strToFloat, y))

test_ids = test["GN/LS"]

RFC = ensemble.RandomForestClassifier(n_estimators = 30, n_jobs = -1)

RFC.fit(X_train, y)
