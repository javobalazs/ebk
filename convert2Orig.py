# Packages
import numpy as np
import math
import cmath
from numpy.core.records import array

# Egy prediktalt polaris diffdiff utvonalbol
#   es az utolso lepes szogebol eloallitja
#   a polaris diff utvonalat


def polarDiff(predLength, predAngle, lastAngle):
    numberOfInputSize = predAngle.shape[0]
    resLength = np.empty(numberOfInputSize, dtype=float)
    resAngle = np.empty(numberOfInputSize, dtype=float)

    i = 0
    preva = lastAngle
    while i < numberOfInputSize:
        resLength[i] = predLength[i]
        preva = predAngle[i] + preva
        while preva < (-180):
            preva += 360
        while preva > 180:
            preva -= 360
        resAngle[i] = preva
        i += 1
    return (resLength, resAngle)


def meterDiff(predLength, predAngle):
    numberOfInputSize = predAngle.shape[0]
    resLatM = np.empty(numberOfInputSize, dtype=float)
    resLongM = np.empty(numberOfInputSize, dtype=float)

    i = 0
    while i < numberOfInputSize:
        l = predLength[i]
        a = math.radians(predAngle[i])
        resLatM[i] = l * math.sin(a)
        resLongM[i] = l * math.cos(a)
        i += 1

    return (resLatM, resLongM)


def converLat2Degrees(latPredM):
    return latPredM / 111320.0


def convertLong2Degrees(longPredM, factor):
    return longPredM / 111320.0 / factor


def normAsAbsolute(route, frst):
    numberOfInputSize = route.shape[0]
    routeAbs = np.empty(numberOfInputSize, dtype=float)
    i = 0
    while i < numberOfInputSize:
        routeAbs[i] = route[i] + frst
        i += 1

    return routeAbs

def convert2AbsDegrees(predLength, predAngle, lastAngle, lLat, lLong, fLat, fLong):
  absLength, absAngle = polarDiff(predLength, predAngle, lastAngle)
  latM, longM = meterDiff(absLength, absAngle)
  latDiff = converLat2Degrees(latM)
  factor = math.cos(math.radians(fLat))
  longDiff = convertLong2Degrees(longM, factor)
  lat = normAsAbsolute(latDiff, lLat)
  long = normAsAbsolute(longDiff, lLong)
  return (lat, long)

