
from pandas import read_csv

import gbUtils as gb

# Read input files
longFileName = "routes_EBK_2_orig_long.csv"
long = gb.readFile(longFileName)
#longTable.drop(axis = 1, labels = [10, 11, 12, 13, 14], inplace = True)

latFileName = "routes_EBK_2_orig_lat.csv"
lat = gb.readFile(latFileName)
#latTable.drop(axis = 1, labels = [10, 11, 12, 13, 14], inplace = True)

predicted_path = 5

mo = gb.model_build(lat, long, predicted_path, testSize=0.0005)
print("test set lat dims: ", mo.lat.shape)
print("test set long dims: ", mo.long.shape)

# print(mo.lat)

# mlat = (mo.lat)[0:1, :-predicted_path]
# mlong = (mo.long)[0:1, :-predicted_path]
mlat = (mo.lat)[:, :-predicted_path]
mlong = (mo.long)[:, :-predicted_path]
print("test set lat dims: ", mlat.shape)
print("test set long dims: ", mlong.shape)

s,f,l,a = mo.ll_predict(mlat, mlong)
print("original: ", mlat)
print("predicted s:", s)
print("original: ", mlong)
print("predicted f:", f)

# print("original full:", mo.lat)
# print("predicted full: ", l)
# print("original full:", mo.long)
# print("predicted full: ", a)

mo.save_model("sandbox/valami")


# inputTableSimple = []
# inputTableSimpleAngle = []
# inputTablePair = []
# complexTable = []
# numberOfRows = latTable.shape[0]
# for indexCounter in range(numberOfRows):

#     lat = latTable.iloc[indexCounter]
#     long = longTable.iloc[indexCounter]
#     (length, angle) = convert.convert2DiffPolarInput(lat, long)
#     complex = convert.ConvertPolar2ComplexInput(length, angle)
#     inputDataSimple, inputDataSimpleFlipped, inputDataPair = convert.convert2PairInput(
#         length, angle)
#     indexCounter += 1
#     inputTableSimple.append(inputDataSimple)
#     inputTableSimpleAngle.append(inputDataSimpleFlipped)
#     inputTablePair.append(inputDataPair)
#     complexTable.append(complex)

# # print(np.shape(inputTableSimple))
# # print(np.shape(inputTableSimple))
# # print(np.shape(inputTablePair))

# dataa = np.array(inputTableSimpleAngle)
# X = dataa[:, :-10]
# y = dataa[:, -10]
# # print(np.shape(X))
# # print(np.shape(y))
# # print(data[1000])
# # print(X[1000])
# # print(y[1000])

# # Split input data into train and test dataset based on test size
# testSize = 0.1
# X_train, aX_test, y_train, y_test = train_test_split(
#     X, y, test_size=testSize, random_state=1)

# print(np.shape(X_train))
# print(np.shape(aX_test))
# print(np.shape(y_train))
# # Create and train the model
# amodel = XGBRegressor()
# amodel.fit(X_train, y_train)
# #model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

# # Maked predictions on the test dataset
# y_pred = amodel.predict(aX_test)
# print(y_pred[100])

# datal = np.array(inputTableSimple)
# X = datal[:, :-10]
# y = datal[:, -10]
# # print(np.shape(X))
# # print(np.shape(y))
# # print(data[1000])
# # print(X[1000])
# # print(y[1000])

# # Split input data into train and test dataset based on test size
# testSize = 0.1
# X_train, lX_test, y_train, y_test = train_test_split(
#     X, y, test_size=testSize, random_state=1)

# print(np.shape(X_train))
# print(np.shape(lX_test))
# print(np.shape(y_train))
# # Create and train the model
# lmodel = XGBRegressor()
# lmodel.fit(X_train, y_train)
# #model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)

# # Maked predictions on the test dataset
# y_pred = lmodel.predict(lX_test)
# print(y_pred[100])

# data_orig = datal[4567]
# print(data_orig)

# dl = datal[4567:4568, :-10]
# # print(dl)
# da = dataa[4567:4568, :-10]
# print(da)


# dlpred = lmodel.predict(dl)
# print(dlpred)
# dapred = amodel.predict(da)
# print(dapred)
