# Packages
import numpy as np
import math
import cmath
from numpy.core.records import array


# Data:
# Geo: (lat, long) degree coordinates
# meterAbs (latMA, longMA) meter from the equator, meter from the Greenwich meridian along the longitudinal circle
# meterDiff (latMD, longMD) meter from the first point. A slight distortion due to different logintudinal circles
#   note: we loose one point, the origo from the sample. We have steps now instead of points.
# polarDiff (length, angle) angle is deviation from north clockwise.
# polarDiffDiff (length, angleD) angleD is deviation from the previous angle.
#   note: we loose one step, the first one, because it has no preceeding step.


# --- Geo <-> Absolute Meter ---
def convertLat2meterAbs(lat):
    return lat * 111320.0

def convertLatFromMeterAbs(latM):
    return latM / 111320.0

def convertLong2meterAbs(lat, long):
    n = lat.shape[0]
    longMeterAbs = np.empty(n, dtype=float)
    i = 0
    while i < n:
        longMeterAbs[i] = long[i] * math.cos(math.radians(lat[i])) * 111320.0
        i += 1
    return longMeterAbs

def convertLongFromMeterAbs(lat, longM):
    n = longM.shape[0]
    long = np.empty(n, dtype=float)
    i = 0
    while i < n:
        long[i] = (longM[i] / math.cos(math.radians(lat[i]))) / 111320.0
        i += 1
    return long

# --- Absolute Meter <-> Meter Difference ---

# np.ediff1d pont jo
# def convert2Diff(route):
#     n = route.shape[0] - 1
#     res = np.empty(n, dtype=float)
#     i = 0
#     j = 1
#     rp = route[0]
#     while i < n:
#         rx = route[j]
#         res[i] = rx - rp
#         rp = rx
#         i += 1
#         j += 1
#     return res

def convertFromDiff(route, start_val):
    n = route.shape[0]
    res = np.empty(n, dtype=float)
    i = 0
    while i < n:
        start_val += route[i]
        res[i] = start_val
        i += 1
    return res

# --- Meter Difference <-> Polar Difference ---

def convert2polarDiff(latMD, longMD):
    n = latMD.shape[0]
    length = np.empty(n, dtype=float)
    angle = np.empty(n, dtype=float)

    i = 0
    while i < n:
        a = latMD[i]
        b = longMD[i]
        length[i] = math.sqrt(a*a + b*b)
        # Az eszakhoz kepesti szoget nezzuk itt,
        # azaz a lattitudakat tekintjuk x-nek.
        angle[i] = math.degrees(math.atan2(b, a))
        i += 1
    return (length, angle)


def convertFromPolarDiff(length, angle):
    n = angle.shape[0]
    latM = np.empty(n, dtype=float)
    longM = np.empty(n, dtype=float)
    i = 0
    while i < n:
        l = length[i]
        a = math.radians(angle[i])
        latM[i] = l * math.cos(a)
        longM[i] = l * math.sin(a)
        i += 1
    return (latM, longM)

# --- Polar Difference <-> Polar DiffDiff ---

def convert2polarDiffDiff(length, angle):
    angleD = np.ediff1d(angle)
    n = angleD.shape[0]
    i = 0
    while i < n:
        ai = angleD[i]
        while ai > 180.0:
            ai -= 360.0
        while ai < (-180.0):
            ai += 360.0
        angleD[i] = ai
        i += 1
    return np.delete(length, 0), angleD

# Convert relative (differential) angles to absolute ones.
#   ai is the absolute angle of the last step before the first one in this sequence.


def convertFromPolarDiffDiff(angleD, ai):
    n = angleD.shape[0]
    angle = np.empty(n, dtype=float)
    i = 0
    while i < n:
        ai += angleD[i]
        while ai > 180.0:
            ai -= 360.0
        while ai < (-180.0):
            ai += 360.0
        angle[i] = ai
        i += 1
    return angle

# --- Geo <-> PolarDiffDiff ---

def convert2(lat, long):
    n = lat.shape[0] - 1
    latM = convertLat2meterAbs(lat)
    longM = convertLong2meterAbs(lat, long)
    latMD = np.ediff1d(latM)
    longMD = np.ediff1d(longM)
    length, angle = convert2polarDiff(latMD, longMD)
    lengthD, angleD = convert2polarDiffDiff(length, angle)
    return lengthD, angleD, (latM[n], longM[n], angle[n-2]), angle, latM, longM, latMD, longMD

def convertFrom(lengthD, angleD, param):
    (latM0, longM0, angle0) = param
    n = angleD.shape[0] - 1
    angle_r = convertFromPolarDiffDiff(angleD, angle0)
    latMD_r, longMD_r = convertFromPolarDiff(lengthD, angle_r)
    latM_r = convertFromDiff(latMD_r, latM0)
    longM_r = convertFromDiff(longMD_r, longM0)
    lat_r = convertLatFromMeterAbs(latM_r)
    long_r = convertLongFromMeterAbs(lat_r, longM_r)
    return lat_r, long_r, (latM_r[n], longM_r[n], angle_r[n])

# # --- TESTS ---

# # Variables
# NumberOfInputSteps     = 10
# # NumberOfPredictedSteps = 5

# lat  = np.empty(NumberOfInputSteps, dtype=float)
# long = np.empty(NumberOfInputSteps, dtype=float)

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

# # --- TEST: construction ---

# print("lat---------------------")
# print(lat)
# print("long---------------------")
# print(long)
# latM = convertLat2meterAbs(lat)
# print("latM---------------------")
# print(latM)
# longM = convertLong2meterAbs(lat, long)
# print("longM---------------------")
# print(longM)
# latMD = np.ediff1d(latM)
# print("latMD---------------------")
# print(latMD)
# longMD = np.ediff1d(longM)
# print("longMD---------------------")
# print(longMD)
# length, angle = convert2polarDiff(latMD, longMD)
# print("length---------------------")
# print(length)
# print("angle---------------------")
# print(angle)
# lengthD, angleD = convert2polarDiffDiff(length, angle)
# print("lengthD---------------------")
# print(lengthD)
# print("angleD---------------------")
# print(angleD)

# # --- TEST: reconstruction ---
# angle_r = convertFromPolarDiffDiff(angleD, angle[0])
# print("angle - angle_r---------------------")
# print(np.delete(angle, 0) - angle_r)
# latMD_r, longMD_r = convertFromPolarDiff(lengthD, angle_r)
# # print("latMD_r---------------------")
# # print(latMD_r)
# # print("longMD_r---------------------")
# # print(longMD_r)
# print("latMD - latMD_r---------------------")
# print(np.delete(latMD, 0) - latMD_r)
# print("longMD - longMD_r---------------------")
# print(np.delete(longMD, 0) - longMD_r)
# latM_r = convertFromDiff(latMD_r, latM[1])
# longM_r = convertFromDiff(longMD_r, longM[1])
# # print("latM_r---------------------")
# # print(latM_r)
# # print("longM_r---------------------")
# # print(longM_r)
# print("latM - latM_r---------------------")
# print(np.delete(latM, [0,1]) - latM_r)
# print("longM - longM_r---------------------")
# print(np.delete(longM, [0,1]) - longM_r)
# lat_r = convertLatFromMeterAbs(latM_r)
# print("lat - lat_r---------------------")
# print(np.delete(lat, [0,1]) - lat_r)
# long_r = convertLongFromMeterAbs(lat_r, longM_r)
# print("long - long_r---------------------")
# print(np.delete(long, [0,1]) - long_r)

# # --- TEST: full conversion ---
# lengthD2, angleD2, paramD2 = convert2(lat, long)
# print("lengthD2---------------------")
# print(lengthD2)
# print("angleD2---------------------")
# print(angleD2)
# print("paramD2---------------------")
# print(paramD2)
# print("lengthD - lengthD2---------------------")
# print(lengthD - lengthD2)
# print("angleD - angleD2---------------------")
# print(angleD - angleD2)
# lat_r2, long_r2, param_r2 = convertFrom(lengthD2, angleD2, (latM[1], longM[1], angle[0]))
# print("lat - lat_r2---------------------")
# print(np.delete(lat, [0,1]) - lat_r2)
# print("long - long_r2---------------------")
# print(np.delete(long, [0,1]) - long_r2)
# print("param_r2---------------------")
# print(param_r2)


# def convert2DiffPolarInput(lat, long):
#     numberOfInputSize = lat.shape[0]

#     longOrigo = normToFirstPointAsOrigo(long)
#     factor = new_func(lat)
#     longOrigoMeter = convertLong2Meter(longOrigo, factor)
#     longDiffMeter = meterDiff(longOrigoMeter)
#     latOrigo = normToFirstPointAsOrigo(lat)
#     latOrigoMeter = convertLat2Meter(latOrigo)
#     latDiffMeter = meterDiff(latOrigoMeter)
#     (length, angle) = polarDiff(latDiffMeter, longDiffMeter)
#     res = polarDiffDiff(length, angle)
#     return res


# --- Utility ---

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

