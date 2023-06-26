# Imports: third party
import numpy as np
from scipy.signal import butter, filtfilt, argrelextrema


def lowpass_filter(signal: np.ndarray, order: int, cutoff: int, fs: int):
    """
    Implementation of a lowpass filter.

    :param signal: <np.ndarray> Signal to apply the filter on.
    :param order: <int> The order of the filter.
    :param cutoff: <int> Cutoff (critical) frequency.
    :param fs: <int> signal sampling frequency.

    :return: <np.ndarray> Filttered signal.
    """

    def _butter_filter(order, cutoff, fs, ftype):
        nyq_freq = fs / 2
        normal_cutoff = cutoff / nyq_freq
        return butter(order, normal_cutoff, btype=ftype, analog=False)

    b_pol, a_pol = _butter_filter(order, cutoff, fs, ftype="low")
    return filtfilt(b_pol, a_pol, signal, axis=0)


def beat_to_beat_features(art_vals: np.ndarray, art_time: np.ndarray, fs: int = 120, twin: float = 0.1):
    """
    Function to extract beat-to-beat mean (map), systolic (sbp), and diastolic (dbp)
    blood pressure, heart rate (hr), and pulse pressure (pp) from a continuous
    blood pressure waveform.

    :param art_vals: <np.ndarray> NumPy array with the ABP waveform values.
    :param art_time: <np.ndarray> NumPy array with the time vector of the ABP
                     measurements.
    :param fs: <int> Sampling frequency.
    :param twin: <float> Time window in seconds to look for the original minimum.

    :return: <Dict[str, np.ndarray]> Returns a dictionary with the detection of the
             beat-to-beat bp features.
             The keys of the dictionary are map, sbp, dbp, pp, hr, and time.
    """
    art_filt_2 = lowpass_filter(art_vals, order=1, cutoff=2, fs=fs)
    art_filt_15 = lowpass_filter(art_vals, order=1, cutoff=15, fs=fs)

    min_idxs = argrelextrema(art_filt_2, np.less)[0]
    def_min_idxs = []
    samples = int(fs * twin)
    for idx, _ in enumerate(min_idxs[:-1]):
        try:
            def_min_idxs.append(
                np.argmin(art_filt_15[min_idxs[idx] : min_idxs[idx] + samples])
                + min_idxs[idx],
                )
        except IndexError:
            continue
    def_min_idxs = sorted(set(def_min_idxs))

    hr_vals = []
    map_vals = []
    dbp_vals = []
    sbp_vals = []
    time = []
    for idx, _ in enumerate(def_min_idxs[:-1]):
        map_vals.append(
            np.mean(art_vals[def_min_idxs[idx] : def_min_idxs[idx + 1]]),
        )
        dbp_vals.append(np.min(art_vals[def_min_idxs[idx] : def_min_idxs[idx + 1]]))
        sbp_vals.append(np.max(art_vals[def_min_idxs[idx] : def_min_idxs[idx + 1]]))
        hr_vals.append(
            60 / (art_time[def_min_idxs[idx + 1]] - art_time[def_min_idxs[idx]]),
            )
        time.append(np.mean(art_time[def_min_idxs[idx] : def_min_idxs[idx + 1]]))

    data = {
        "map": np.array(map_vals),
        "sbp": np.array(sbp_vals),
        "dbp": np.array(dbp_vals),
        "pp": np.array(sbp_vals) - np.array(dbp_vals),
        "hr": np.array(hr_vals),
        "time": np.array(time),
    }
    return data
