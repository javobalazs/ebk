# Packages
import numpy as np
import math
import cmath
from numpy.core.records import array

# # Variables
# NumberOfInputSteps     = 10
# NumberOfPredictedSteps = 5
#
# lat  = np.empty(NumberOfInputSteps, dtype=float)
# long = np.empty(NumberOfInputSteps, dtype=float)
#
# lat[0] = 47.5945153413324
# lat[1] = 47.5945156826648
# lat[2] = 47.5945160239973
# lat[3] = 47.5945163653297
# lat[4] = 47.5945167066621
# lat[5] = 47.5945170479946
# lat[6] = 47.594517389327
# lat[7] = 47.5945177306594
# lat[8] = 47.5945180719918
# lat[9] = 47.5945184133243
# long[0] = 19.3608201706662
# long[1] = 19.3608203413324
# long[2] = 19.3608205119986
# long[3] = 19.3608206826649
# long[4] = 19.3608208533311
# long[5] = 19.3608210239973
# long[6] = 19.3608211946635
# long[7] = 19.3608213653297
# long[8] = 19.3608215359959
# long[9] = 19.3608217066621


def normToFirstPointAsOrigo(route):
    numberOfInputSize = route.shape[0]
    numberOfDiffSize = numberOfInputSize - 1

    routeDiff = np.empty(numberOfDiffSize, dtype=float)
    initial = route[0]

    i = 1
    while i < numberOfInputSize:
        routeDiff[i - 1] = route[i] - initial
        i += 1

    return routeDiff


def convertLat2Meter(latOri):
    return latOri*111320.0


def convertLong2Meter(longOri, factor):
    return longOri * factor * 111320.0


def meterDiff(route):
    numberOfInputSize = route.shape[0]
    routeDiff = np.empty(numberOfInputSize, dtype=float)

    routeDiff[0] = route[0]
    i = 1
    while i < numberOfInputSize:
        routeDiff[i] = route[i] - route[i-1]
        i += 1
    return routeDiff


def polarDiff(lat, long):
    numberOfInputSize = lat.shape[0]
    length = np.empty(numberOfInputSize, dtype=float)
    angle = np.empty(numberOfInputSize, dtype=float)

    i = 0
    while i < numberOfInputSize:
        a = lat[i]
        b = long[i]
        length[i] = math.sqrt(a*a + b*b)
        # Az eszakhoz kepesti szoget nezzuk itt,
        # azaz a lattitudakat tekintjuk x-nek.
        angle[i] = math.degrees(math.atan2(b, a))
        i += 1
    return (length, angle)


def polarDiffDiff(length, angle):
    numberOfInputSize = angle.shape[0]
    resLength = np.empty(numberOfInputSize - 1, dtype=float)
    resAngle = np.empty(numberOfInputSize - 1, dtype=float)

    i = 1
    j = 0
    while i < numberOfInputSize:
        resLength[j] = length[i]
        resAngle_j = angle[i] - angle[j]
        while resAngle_j < (-180.0):
            resAngle_j += 360.0
        while resAngle_j > 180.0:
            resAngle_j -= 360.0
        resAngle[j] = resAngle_j
        i += 1
        j += 1
    return (resLength, resAngle)


def convert2DiffPolarInput(lat, long):
    numberOfInputSize = lat.shape[0]

    longOrigo = normToFirstPointAsOrigo(long)
    factor = new_func(lat)
    longOrigoMeter = convertLong2Meter(longOrigo, factor)
    longDiffMeter = meterDiff(longOrigoMeter)
    latOrigo = normToFirstPointAsOrigo(lat)
    latOrigoMeter = convertLat2Meter(latOrigo)
    latDiffMeter = meterDiff(latOrigoMeter)
    (length, angle) = polarDiff(latDiffMeter, longDiffMeter)
    res = polarDiffDiff(length, angle)

    return res

def new_func(lat):
    math.cos(math.radians(lat[0]))


def ConvertPolar2ComplexInput(length, angle):
    numberOfInputSize = length.shape[0]
    cpl = np.empty(numberOfInputSize, dtype=complex)
    i = 0
    while i < numberOfInputSize:
        cpl[i] = cmath.rect(length[i], math.radians(angle[i]))
        i += 1

    return cpl


def convert2PairInput(length, angle):
    dataPair = []
    dataSimple = []
    dataSimpleFlipped = []
    for index in range(len(length)):
        pair = np.empty(2, dtype=float)
        pair[0] = length[index]
        pair[1] = angle[index]
        dataPair.append(pair)
        dataSimple.append(length[index])
        dataSimple.append(angle[index])
        dataSimpleFlipped.append(angle[index])
        dataSimpleFlipped.append(length[index])
    return dataSimple, dataSimpleFlipped, dataPair

# # Main program
# (length, angle) = def convert2DiffPolarInput(lat, long)
# complex = ConvertPolar2ComplexInput
# inputDataSimple, inputDataPair = convert2PairInput(length, angle)
#
# # Test output
# print("lenght")
# print(length)
# print("---------------------")
# print("angle")
# print(angle)
# print("---------------------")
# print("complex")
# print(complex)
# print("---------------------")
# print("simple")
# print(inputDataSimple)
# print("---------------------")
# print("pair")
# print(inputDataPair)
# print("---------------------")
