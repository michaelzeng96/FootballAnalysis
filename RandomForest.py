from sklearn import ensemble
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

# read in and format data
def strToFloat(data):
    stringNums = {string: i for i, string in enumerate(set(data))}
    return np.array([stringNums[d] for d in data])

raw_df = pd.read_csv("data.csv")
df = pd.DataFrame()

# convert strings to integer classes
for col in raw_df.columns:
    if raw_df[col].dtype == np.dtype('O'):
        df[col] = strToFloat(raw_df[col])
    else:
        df[col] = raw_df[col]

del raw_df

# replace NaNs (missing values) with -1
df[np.isnan(df)] = -1

train, test = train_test_split(df, test_size=0.2)

y_train = train["GN/LS"]
x_train = train
del x_train["GN/LS"]

y_test = test["GN/LS"]
x_test = test
del x_test["GN/LS"]

RFC = ensemble.RandomForestClassifier(n_estimators=30, n_jobs=-1)
RFC.fit(x_train, y_train)

predicted = RFC.predict(x_test)
actual = np.array(y_test)
correlation = np.corrcoef(np.array([predicted, actual]))[0, 1]
print("Correlation: ", correlation)