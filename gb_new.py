import convert2Input
from numpy.core.fromnumeric import shape
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from numpy import asarray
from pandas import read_csv
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import convert2Input as convert


def readFile(fName):
    tab = pd.read_csv(fName, encoding="ISO-8859-1", header=None, sep=',')
    tab.astype('float')
    return tab


def readFile(fName):
    tab = pd.read_csv(fName, encoding="ISO-8859-1", header=None, sep=';')
    tab.astype('float')
    return tab


# Read input files
longFileName = "/media/sf_EBK/input/routes_EBK_2_orig_long.csv"
longTable = readFile(longFileName)
#longTable.drop(axis = 1, labels = [10, 11, 12, 13, 14], inplace = True)

latFileName = "/media/sf_EBK/input/routes_EBK_2_orig_lat.csv"
latTable = readFile(latFileName)
#latTable.drop(axis = 1, labels = [10, 11, 12, 13, 14], inplace = True)

inputTableSimple = []
inputTableSimpleAngle = []
inputTablePair = []
complexTable = []
numberOfRows = latTable.shape[0]
for indexCounter in range(numberOfRows):

    lat = latTable.iloc[indexCounter]
    long = longTable.iloc[indexCounter]
    (length, angle) = convert.convert2DiffPolarInput(lat, long)
    complex = convert.ConvertPolar2ComplexInput(length, angle)
    inputDataSimple, inputDataSimpleFlipped, inputDataPair = convert.convert2PairInput(
        length, angle)
    indexCounter += 1
    inputTableSimple.append(inputDataSimple)
    inputTableSimpleAngle.append(inputDataSimpleFlipped)
    inputTablePair.append(inputDataPair)
    complexTable.append(complex)

# print(np.shape(inputTableSimple))
# print(np.shape(inputTableSimple))
# print(np.shape(inputTablePair))

dataa = np.array(inputTableSimpleAngle)
X = dataa[:, :-10]
y = dataa[:, -10]
# print(np.shape(X))
# print(np.shape(y))
# print(data[1000])
# print(X[1000])
# print(y[1000])

# Split input data into train and test dataset based on test size
testSize = 0.1
X_train, aX_test, y_train, y_test = train_test_split(
    X, y, test_size=testSize, random_state=1)

print(np.shape(X_train))
print(np.shape(aX_test))
print(np.shape(y_train))
# Create and train the model
amodel = XGBRegressor()
amodel.fit(X_train, y_train)
#model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

# Maked predictions on the test dataset
y_pred = amodel.predict(aX_test)
print(y_pred[100])

datal = np.array(inputTableSimple)
X = datal[:, :-10]
y = datal[:, -10]
# print(np.shape(X))
# print(np.shape(y))
# print(data[1000])
# print(X[1000])
# print(y[1000])

# Split input data into train and test dataset based on test size
testSize = 0.1
X_train, lX_test, y_train, y_test = train_test_split(
    X, y, test_size=testSize, random_state=1)

print(np.shape(X_train))
print(np.shape(lX_test))
print(np.shape(y_train))
# Create and train the model
lmodel = XGBRegressor()
lmodel.fit(X_train, y_train)
#model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

# Maked predictions on the test dataset
y_pred = lmodel.predict(lX_test)
print(y_pred[100])

data_orig = datal[4567]
print(data_orig)

dl = datal[4567:4568, :-10]
# print(dl)
da = dataa[4567:4568, :-10]
print(da)


dlpred = lmodel.predict(dl)
print(dlpred)
dapred = amodel.predict(da)
print(dapred)
