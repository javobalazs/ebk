from numpy.core.fromnumeric import shape
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import convertUtils as cv


def convert2lifted(lat, long):
    rows = lat.shape[0]
    cols = lat.shape[1]
    straight = np.empty([rows, (cols-2)*2], dtype=float)
    flipped = np.empty([rows, (cols-2)*2], dtype=float)
    length_r = np.empty([rows, cols-2], dtype=float)
    angled_r = np.empty([rows, cols-2], dtype=float)
    angle_r = np.empty([rows, cols-1], dtype=float)
    latM_r = np.empty([rows, cols], dtype=float)
    longM_r = np.empty([rows, cols], dtype=float)
    latMD_r = np.empty([rows, cols-1], dtype=float)
    longMD_r = np.empty([rows, cols-1], dtype=float)
    params = np.empty([rows, 3], dtype=float)
    for i in range(rows):
        (length, angled, param, angle, latM, longM, latMD, longMD) = cv.convert2(lat[i], long[i])
        straight[i], flipped[i], x = cv.convert2PairInput(length, angled)
        params[i] = param
        length_r[i] = length
        angled_r[i] = angled
        angle_r[i] = angle
        latM_r[i] = latM
        longM_r[i] = longM
        latMD_r[i] = latMD
        longMD_r[i] = longMD
    return straight, flipped, length_r, angled_r, params, angle_r, latM_r, longM_r, latMD_r, longMD_r


# predictedPathLength: hany lepest akarunk tippelni
def generate_training_data(data, predictedPathLength, testSize=0.1, randomState=1, verbose=True):
    cols = data.shape[1]
    diff = predictedPathLength * 2
    res = []
    while diff >= 2:
        j = diff
        a = 0
        X_train_i = []
        Y_train_i = []
        path = int((cols - diff)/2)
        while j >= 2:
            if verbose:
                print("generate_training_data, path: ",path, ", displacement: ", int(a/2))
            X = data[:, a:-j]
            Y = data[:, -j]
            X_train_i.append(X)
            Y_train_i.append(Y)
            j -= 2
            a += 2
        X_train = np.concatenate(X_train_i)
        Y_train = np.concatenate(Y_train_i)
        if verbose:
            print("generate_training_data, train_rows: ", X_train.shape[0], ", cols: ", X_train.shape[1])
        res.append([path, X_train, Y_train])
        diff -= 2
    return res


def train(training_data, verbose=True):
    l = len(training_data)
    i = 0
    res = []
    while i < l:
        dsc = training_data[i]
        if verbose:
            print("train, path: ", dsc[0], ", train_rows: ", dsc[1].shape[0])
        model = XGBRegressor()
        model.fit(dsc[1], dsc[2], verbose=verbose)
        res.append([dsc[0], model])
        i += 1
    return res


class gbUtils:
    def __init__(self, straight, flipped, lat, long, length, angled, params, angle, latM, longM, latMD, longMD, model_straight, model_flipped):
        self.straight = straight
        self.flipped = flipped
        self.length = length
        self.angled = angled
        self.params = params
        self.angle = angle
        self.lat = lat
        self.long = long
        self.latM = latM
        self.longM = longM
        self.latMD = latMD
        self.longMD = longMD
        self.model_flipped = model_flipped
        self.model_straight = model_straight

    def predict(self, straight, flipped, verbose=True):
        rows = straight.shape[0]
        path = straight.shape[1]
        p = int(path/2)
        if verbose:
            print("predict: rows:", rows, ", path: ", p)
        i = 0
        rstraight = np.copy(straight)
        rflipped = np.copy(flipped)
        angled = []
        length = []
        while i < len(self.model_straight):
            ms = self.model_straight[i]
            mf = self.model_flipped[i]
            if p == ms[0]:
                if verbose:
                    print("predict for path: ", p)
                ps = ms[1].predict(rstraight)
                ps = np.reshape(ps, [rows, 1])
                pf = mf[1].predict(rflipped)
                pf = np.reshape(pf, [rows, 1])
                rstraight = np.concatenate([rstraight, ps, pf], axis=1)
                rflipped = np.concatenate([rflipped, pf, ps], axis=1)
                angled.append(pf)
                length.append(ps)
            i += 1
            p += 1
        return rstraight, rflipped, np.concatenate(length, axis=1), np.concatenate(angled, axis=1)

    def ll_predict(self, lat, long, verbose=True):
        rows = lat.shape[0]
        path = lat.shape[1]
        if verbose:
            print("ll_predict: rows:", rows, ", points: ", path)
        straight = np.empty([rows, (path - 2)*2], dtype=float)
        flipped = np.empty([rows, (path - 2)*2], dtype=float)
        params = np.empty([rows, 3], dtype=float)
        i = 0
        while i < rows:
            lengthD, angleD, param, angle, latM, longM, latMD, longMD = cv.convert2(lat[i], long[i])
            straight[i], flipped[i], x = cv.convert2PairInput(lengthD, angleD)
            params[i] = param
            i += 1
        rstraight, rflipped, pred_l, pred_a = self.predict(straight, flipped, verbose)
        i = 0
        pred_lat_r = []
        pred_long_r = []
        while i < rows:
            # pred_lat, pred_long, x = cv.convertFrom(pred_l[i], pred_a[i], (param[i,0], param[i,1], param[i,2]))
            pred_lat, pred_long, x = cv.convertFrom(pred_l[i], pred_a[i], params[i])
            pred_lat = np.reshape(pred_lat, [1, pred_lat.shape[0]])
            pred_long = np.reshape(pred_long, [1, pred_long.shape[0]])

            pred_lat_r.append(pred_lat)
            pred_long_r.append(pred_long)
            i += 1
        platr = np.concatenate(pred_lat_r)
        plongr = np.concatenate(pred_long_r)
        return platr, plongr,platr, plongr

def model_build(lat, long, predictedPathLength, verbose=True, randomState=1, testSize=0.1):
    lat = np.array(lat)
    long = np.array(long)
    if verbose:
        print("model_build, predictedPathLength: ", predictedPathLength)
        print("model_build, lat, rows: ",
              lat.shape[0], ", steps: ", lat.shape[1])
        print("model_build, long, rows: ",
              long.shape[0], ", steps: ", long.shape[1])
    straight, flipped, length, angled, params, angle, latM, longM, latMD, longMD = convert2lifted(lat, long)
    if verbose:
        print("model_build, params: ", params.shape)
        print("model_build, straight, rows: ",
              straight.shape[0], ", cols: ", straight.shape[1])
        print("model_build, flipped, rows: ",
              flipped.shape[0], ", cols: ", flipped.shape[1])
    straight_train, straight_test, flipped_train, flipped_test, lat_train, lat_test, long_train, long_test, length_train, length_test, angled_train, angled_test, params_train, params_test, angle_train, angle_test, latM_train, latM_test, longM_train, longM_test, latMD_train, latMD_test, longMD_train, longMD_test = train_test_split(
        straight, flipped, lat, long, length, angled, params, angle, latM, longM, latMD, longMD, test_size=testSize, random_state=randomState)
    if verbose:
        print("test set lat rows: ", lat_test.shape[0], ", cols: ", lat_test.shape[1])
        print("test set long rows: ", long_test.shape[0], ", cols: ", long_test.shape[1])
    if verbose:
        print("training_data for straight (length first)")
    trainingset_straight = generate_training_data(straight_train, predictedPathLength, verbose=True, randomState=randomState, testSize=testSize)
    if verbose:
        print("training_data for flipped (angle first)")
    trainingset_flipped = generate_training_data(flipped_train, predictedPathLength, verbose=True, randomState=randomState, testSize=testSize)
    if verbose:
        print("train for straight (length first)")
    model_straight = train(trainingset_straight, verbose=verbose)
    if verbose:
        print("train for flipped (angle first)")
    model_flipped = train(trainingset_flipped, verbose=verbose)
    return gbUtils(straight_test, flipped_test, lat_test, long_test, length_test, angled_test, params_test, angle_test, latM_test, longM_test, latMD_test, longMD_test, model_straight, model_flipped)
    # return gbUtils(straight_test, flipped_test, lat_test, long_test, length_test, angled_test, params_test, angle_test, latM_test, longM_test, latMD_test, longMD_test, 1, 1)


def readFile(fName):
    tab = pd.read_csv(fName, encoding="ISO-8859-1", header=None, sep=',')
    tab.astype('float')
    return tab


def readFile(fName):
    tab = pd.read_csv(fName, encoding="ISO-8859-1", header=None, sep=';')
    tab.astype('float')
    return tab


