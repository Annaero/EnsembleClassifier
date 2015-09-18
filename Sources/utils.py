# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 19:07:23 2015

@author: Annaero
"""

from itertools import islice
from math import sqrt
from itertools import product
from dtw import dtw as lib_dtw

from numpy.linalg import lstsq
from sklearn.metrics import mean_squared_error

#Reciept from itertools documentation
def window(seq, n=2, step=1):
    """ Returns a sliding window (of width n) over data from the iterable
        s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        print(result[step:])
        result = result[step:] + (elem,)
        yield result

def rmse(actual, predicted):
    ln = min(len(predicted), len(actual))
    rmse = sqrt(mean_squared_error(actual[:ln], predicted[:ln]))
    return rmse
    
def dtw(predicted, actual):
    ln = min(len(predicted), len(actual))
    dist, cost, path = lib_dtw(predicted[:ln], actual[:ln])
    return dist

def calibrate_ensemble(models_forecasts, measurements, forecast_len = 48):
    """Calculates coefficient for models in ensembles usulg OLS.
       Returns coefficients for all possible ensembles obtained by models combinations.
    """
    models_count = len(models_forecasts)

    predictors = [list() for mdl in range(models_count)]
    target = list()

    rng = min(len(model) for model in models_forecasts)
    for tm in range(rng):
        msm = measurements[tm * 6: tm * 6 + forecast_len]
        msm_len = len(msm)
        for current_prediction, predictor \
                in zip([prd[tm] for prd in models_forecasts], predictors):
            predictor.extend(current_prediction[:msm_len])
        target.extend(msm)

    ensembles = list()
    for ens_map in reversed(list(product([1,0], repeat = models_count))):
        ensemble_predictors = \
                    [[a*b for a,b in zip(point, ens_map)] for point in zip(*predictors)]

        ensemble_predictors = [pred + [1] for pred in ensemble_predictors]
        coefs = list(lstsq(ensemble_predictors, target)[0])
        ensembles.append(coefs)
    return ensembles

def find_corresponding_peak(peak, peaks):
#    absolute_peak_coords = peak[0] + peak[2]
    corresponding = [p for p in peaks if peak[0] <= p[2] <= peak[1]]
    if not corresponding:
        return None
    nearest = sorted(corresponding, key = lambda x: abs(x[2] - peak[2]))[0]
    return nearest

def calibrate_peak_ensemble(models_forecasts, measurements, forecast_len = 48, peak_level = 80):
    T_predictors = []
    T_target = []
    H_predictors = []
    H_target = []

    rng = min(len(model) for model in models_forecasts)
    for tm in range(rng):
        msm = measurements[tm * 6: tm * 6 + forecast_len]
        measured_peaks = detect_peaks(msm, peak_level)
        if not measured_peaks:
            continue

        forecasts = [prd[tm] for prd in models_forecasts]
        forecasts_peaks = [detect_peaks(fcst, peak_level) for fcst in forecasts]
        
        forecasts_peaks_cor = [list(map(lambda x: find_corresponding_peak(x, forecast_peaks),
                                      measured_peaks)) for forecast_peaks in forecasts_peaks]
        for measured, *corresponding in zip(measured_peaks, *forecasts_peaks_cor):
            if all(corresponding):
                H_predictors.append([peak[3] for peak in corresponding] + [1])
                T_predictors.append([peak[2] for peak in corresponding] + [1])
                H_target.append(measured[3])
                T_target.append(measured[2])

    print(H_predictors, H_target)

    H_coefs = lstsq(H_predictors, H_target)[0]
    T_coefs = lstsq(T_predictors, T_target)[0]
    return list(T_coefs), list(H_coefs)

def detect_peaks(forecast, peak_level = 80):
    """Finds all peaks in time series and return theirs parameters
    Returns list of tuples (u, d, H, T) where u, d, T is forecast point number
    from the beginning of forecast for beginning of the peak, its end, and its
    Heights point respectively. H is the value at the highest point of the peak.
    """
    forecast_len = len(forecast)
    peaks = []

    #Finding level intersection
    intersections = []
    for i in range(1, forecast_len):
        if forecast[i-1] <= peak_level and forecast[i] > peak_level:
            intersections.append(("u", i))
        elif forecast[i-1] > peak_level and forecast[i] <= peak_level and \
            intersections and intersections[-1][0] == "u":
            intersections.append(("d", i))

    if not intersections:
        return peaks

    if len(intersections) % 2 != 0 and intersections[-1][0] == "u":
        intersections = intersections[:-1]

    intersections = [i[1] for i in intersections]

    #Finding peaks H and T
    for u, d in zip(intersections[::2], intersections[1::2]):
        H = max(forecast[u:d])
        T = forecast[u:d].index(H) + u
        peaks.append((u, d, T, H))

    return peaks

def correct_forecast(ensemble_forecast, models_forecasts, 
                     peak_ensemble_coeffs, peak_level = 80):
    ensemble_peaks = detect_peaks(ensemble_forecast, peak_level)
    if not ensemble_peaks:
        ensemble_forecast

    forecasts_peaks = [detect_peaks(fcst, peak_level) for fcst in models_forecasts]
    forecasts_peaks_cor = [list(map(lambda x: find_corresponding_peak(x, forecast_peaks),
                              ensemble_peaks)) for forecast_peaks in forecasts_peaks]
                              
    correctors = []
    for measured, *corresponding in zip(ensemble_peaks, *forecasts_peaks_cor):
#        print(corresponding)
        if all(corresponding):
            T, H = _peak_ensemble(corresponding, peak_ensemble_coeffs)
            correctors.append((T,H))
        else:
            correctors.append((None, None))
#    print(correctors)
            
    corrected_forecast = correct_forecast_peaks(ensemble_forecast, correctors)
    return corrected_forecast

def _peak_ensemble(peaks, coeffs):
    rH, rT = 0, 0
    for peak, t_coeff, h_coeff  in zip(peaks, *coeffs):
        u, d, T, H = peak
        rH += H * h_coeff
        rT += T * t_coeff
    rH+=coeffs[1][-1]
    rT+=coeffs[0][-1]
    return rT, rH
    
def correct_forecast_peaks(ensemble_forecast, correctors, peak_level = 80):
    ensemble_peaks = detect_peaks(ensemble_forecast, peak_level)
    corrected_forecast = list(ensemble_forecast)
    for (u, d, eT, eH), (T, H) in zip(ensemble_peaks, correctors):
        if not (T and H):
            continue
        h_corrector = H/eH
        t_corrector = T/eT
        peak = ensemble_forecast[u:d]
        corrected_levels = list(map(lambda x : x * h_corrector, peak))
#        print(list(corrected_levels))

        times = range(u, d)# + 1)
        corrected_times = map(lambda x : x * t_corrector, times)

        corrected_peak = []
        points = list(zip(corrected_times, corrected_levels))
        for time in times:
            level = _find_corrected_level(time, points)
            corrected_peak.append(level)
        corrected_forecast[u:d] = corrected_peak
    return corrected_forecast


    
def _find_corrected_level(time, points):
    for c_time, c_level in points:
        if c_time == time:
            return c_level
        elif c_time < time:
            continue
        idx = points.index((c_time, c_level))
        if idx == 0:
            return c_level
        c_time1, c_level1 = points[idx-1]
        return c_level+(time-c_time)*(c_level1-c_level)/(c_time1-c_time)
    return points[-1][1]