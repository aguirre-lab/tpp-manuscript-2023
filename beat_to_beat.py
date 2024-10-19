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


def beat_to_beat_features(art_vals: np.ndarray, art_time: np.ndarray, fs: int = 120, twin: float = 0.1, by='max', filt_outliers=False):
    """
    Function to extract beat-to-beat mean (map), systolic (sbp), and diastolic (dbp)
    blood pressure, heart rate (hr), and pulse pressure (pp) from a continuous
    blood pressure waveform.

    :param art_vals: <np.ndarray> NumPy array with the ABP waveform values.
    :param art_time: <np.ndarray> NumPy array with the time vector of the ABP
                     measurements.
    :param fs: <int> Sampling frequency.
    :param twin: <float> Time window in seconds to look for the original minimum.
    :param by: <str> Extrema to use to segment cardiac cycles. Either min or max.
    :param filt_outliers: <bool> Use an outlier filter. This may slow down the processing time.

    :return: <Dict[str, np.ndarray]> Returns a dictionary with the detection of the
             beat-to-beat bp features.
             The keys of the dictionary are map, sbp, dbp, pp, hr, and time.
    """
    if by == 'max':
        op1 = np.greater
        op2 = np.argmax
        cutoff = 0.75
    elif by == 'min':
        op1 = np.less
        op2 = np.argmin
        cutoff = 1.5
        
    art_filt_2 = lowpass_filter(art_vals, order=1, cutoff=cutoff, fs=fs)
    art_filt_15 = lowpass_filter(art_vals, order=1, cutoff=15, fs=fs)

    ext_idxs = argrelextrema(art_filt_2, op1)[0]
    def_ext_idxs = []
    samples = int(fs * twin)
    for idx, _ in enumerate(ext_idxs[:-1]):
        try:
            def_ext_idxs.append(
                op2(art_filt_15[ext_idxs[idx] : ext_idxs[idx] + samples])
                + ext_idxs[idx],
                )
        except IndexError:
            continue
    def_ext_idxs = sorted(set(def_ext_idxs))

    hr_vals = []
    map_vals = []
    dbp_vals = []
    sbp_vals = []
    time = []
    for idx, _ in enumerate(def_ext_idxs[:-1]):
        map_vals.append(
            np.mean(art_vals[def_ext_idxs[idx] : def_ext_idxs[idx + 1]]),
        )
        dbp_vals.append(np.min(art_vals[def_ext_idxs[idx] : def_ext_idxs[idx + 1]]))
        sbp_vals.append(np.max(art_vals[def_ext_idxs[idx] : def_ext_idxs[idx + 1]]))
        hr_vals.append(
            60 / (art_time[def_ext_idxs[idx + 1]] - art_time[def_ext_idxs[idx]]),
            )
        time.append(np.mean(art_time[def_ext_idxs[idx] : def_ext_idxs[idx + 1]]))

    data = {
        "map": np.array(map_vals),
        "sbp": np.array(sbp_vals),
        "dbp": np.array(dbp_vals),
        "pp": np.array(sbp_vals) - np.array(dbp_vals),
        "hr": np.array(hr_vals),
        "time": np.array(time),
    }
    
    if filt_outliers:
        def _remove_outliers(data, wins, thresh):
            for el, arr in data.items():
                # Create arrays for rolling mean and std, handling edges by reflecting the data
                rolling_mean = np.convolve(arr, np.ones(wins)/wins, mode='same')
                rolling_std = np.sqrt(np.convolve((arr - rolling_mean)**2, np.ones(wins)/wins, mode='same'))
    
                # Identify outliers based on deviation from the rolling mean
                outliers_mask = np.abs(arr - rolling_mean) > thresh * rolling_std

                # Loop through the array and replace outliers with the average of their nearest valid neighbors
                for idx in range(len(arr)):
                    if outliers_mask[idx]:
                        # Find valid neighbors
                        left = arr[:idx][~outliers_mask[:idx]]
                        right = arr[idx+1:][~outliers_mask[idx+1:]]
                        neighbors = np.concatenate([left[-1:], right[:1]])  # Get closest valid neighbors
                        if len(neighbors) > 0:
                            arr[idx] = np.mean(neighbors)
                data[el] = arr
            return data
        data = _remove_outliers(data, wins=20, thresh=0.5)
    return data
