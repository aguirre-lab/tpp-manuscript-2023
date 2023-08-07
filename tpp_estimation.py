# Imports: standard library
from typing import Dict, List

# Imports: third party
import numpy as np
from sklearn.linear_model import LinearRegression

# Imports: first party
from beat_to_beat import beat_to_beat_features


def pcrit_estimation(maps: np.ndarray, pps_hrs: np.ndarray):
    """
    Estimate critical closing pressure given a set of MAP and PP*HR paired data points.
    To estimate critical closing pressure, the intercept of the cloud of points formed
    by the MAP vs PP*HR plot is used. That is, an estimation of the pressure when there
    is 0 flow. To remove outliers, just points between the 5 and 95 percentil of
    PP*HR are used.

    :param maps: <np.ndarray> Set of MAP data points.
    :param pps_hrs: <np.ndarray> Set of PP*HR data points.

    :return: <Tuple[Float, Float, Float, np.ndarray]> Returns a Tuple with the estimated
             critical closing pressure, the slope and the r2 square of the linear fit,
             and a NumPy array with the indices of the data points used for the
             estimation.
    """
    no_nans = np.logical_and(~np.isnan(pps_hrs), ~np.isnan(maps))
    no_inf = np.logical_and(~np.isinf(pps_hrs), ~np.isinf(maps))
    indices = np.where(np.logical_and(no_nans, no_inf))[0]

    if len(indices) >= 5:
        percentil = np.percentile(pps_hrs[indices], [5, 95])
        no_outliers = np.argwhere(
            np.logical_and(
                pps_hrs[indices] >= percentil[0],
                pps_hrs[indices] <= percentil[1],
            ),
        ).reshape(-1)
        indices = indices[no_outliers]

    if len(indices) > 1:
        regresion = LinearRegression().fit(
            pps_hrs[indices].reshape(-1, 1),
            maps[indices].reshape(-1),
        )
        pcrit = regresion.intercept_
        slope = regresion.coef_[0]
        r2 = regresion.score(
            pps_hrs[indices].reshape(-1, 1),
            maps[indices].reshape(-1),
        )
    else:
        pcrit = np.nan
        slope = np.nan
        r2 = np.nan

    return pcrit, slope, r2, indices


def sliding_window_pcrit_estimation_from_features(
    maps: np.ndarray,
    pps: np.ndarray,
    hrs: np.ndarray,
    time: np.ndarray,
    window: int = 1,
    step: int = 1,
):
    """
    Estimate critical closing pressure for a continuous set of beat-to-beat values of
    mean arterial pressure, pulse pressure, and heart rate using a sliding window
    approach.

    :param maps: <np.ndarray> Value array of MAP values.
    :param pps: <np.ndarray> Value array with PP values.
    :param hrs: <np.ndarray> Value array with hr values.
    :param time: <np.ndarray> Time array in seconds.
    :param window: <int> Look-back window size in minutes.
    :param step: <int> Sliding window step in minutes.

    :return: <Dict[str, np.ndarray]> Returns a dictionary with the estimation of
             critical closing pressure (pcrit), tissue perfusion pressure (tpp),
             and the average map, pp, hr, pp_hr of the window using a sliding
             window of size `window` minutes and a step of `step` minutes.
             It also provides the slope and the r2 of every fit.
             The keys of the dictionary are pcrit, tpp, slope, r2, map, pp, hr, pp_hr,
             and time.
    """
    sliding_window = window * 60
    init = step * 60
    data: Dict[str, List[float]] = {
        "pcrit": [],
        "tpp": [],
        "slope": [],
        "r2": [],
        "map": [],
        "pp": [],
        "hr": [],
        "pp_hr": [],
        "time": [],
    }
    pp_hr = pps * hrs
    relative_time = time - time[0]
    while init - sliding_window < relative_time[-1]:
        time_indices = np.logical_and(
            relative_time > init - sliding_window,
            relative_time <= init,
        )
        if not time_indices.any():
            init = relative_time[np.where(relative_time > init)[0][0]]
        else:
            init += step * 60

            pcrit, slope, r2, indices = pcrit_estimation(
                pp_hr[time_indices],
                maps[time_indices],
            )

            data["pcrit"].append(pcrit)
            data["tpp"].append(pcrit - np.mean(maps[time_indices][indices]))
            data["slope"].append(slope)
            data["r2"].append(r2)
            data["map"].append(np.mean(maps[time_indices][indices]))
            data["pp"].append(np.mean(pps[time_indices][indices]))
            data["hr"].append(np.mean(hrs[time_indices][indices]))
            data["pp_hr"].append(np.mean(pp_hr[time_indices][indices]))
            data["time"].append(init + time[0])

    # Convert output dict items into NumPy arrays
    for signal in data:
        data[signal] = np.array(data[signal])

    return data


def sliding_window_pcrit_estimation_from_waveform(
    art_vals: np.ndarray,
    art_time: np.ndarray,
    fs: int = 120,
    window: int = 1,
    step: int = 1,
):
    """
    Estimate critical closing pressure for a continuous blood pressure waveform
    using a sliding window approach.

    :param art_vals: <np.ndarray> NumPy array with the ABP waveform values.
    :param art_time: <np.ndarray> NumPy array with the time vector of the ABP
                      measurements.
    :param fs: <int> Sampling frequency.
    :param window: <int> Look-back window size in minutes.
    :param step: <int> Sliding window step in minutes.

    :return: <Dict[str, np.ndarray]> Returns a dictionary with the estimation of
             critical closing pressure (pcrit), tissue perfusion pressure (tpp),
             and the average map, pp, hr, pp_hr of the window using a sliding
             window of size `window` minutes and a step of `step` minutes.
             It also provides the slope and the r2 of every fit.
             The keys of the dictionary are pcrit, tpp, slope, r2, map, pp, hr, pp_hr,
             and time.
    """
    bp_features = beat_to_beat_features(art_vals, art_time, fs)
    data = sliding_window_pcrit_estimation_from_features(
        maps=bp_features["map"],
        pps=bp_features["pp"],
        hrs=bp_features["hr"],
        time=bp_features["time"],
        window=window,
        step=step,
    )
    return data
