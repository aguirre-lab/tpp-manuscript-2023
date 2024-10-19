# Tissue Perfusion Pressure Estimation
Code to estimate tissue perfusion pressure from a high-frequency arterial line waveform. Companion code to the paper “Tissue perfusion pressure enables continuous hemodynamic evaluation and risk prediction in the intensive care unit”.

The code in this repository is made available for the purposes of scientific research and educational activities in conjunction with the corresponding published scientific manuscript.

The methods presented and the code provided are not intended for and should not be utilized for commercial purposes or for clinical evaluation, clinical decision making, or healthcare delivery. The user holds the authors and their corresponding institutions harmless from any claims arising out of its use. The authors and their corresponding institutions are not responsible for any treatment or medical applications or decisions made by users based on information contained in this repository. 

A patent application entitled “Systems and Methods for Measuring Critical Closing Pressure and Tissue Perfusion Pressure” was filed by the General Hospital Corporation with the US Patent and Trademark Office. 

Please direct any questions or further inquiries to the corresponding author at aaguirre1@mgh.harvard.edu.

--------
Citation:
```
Chandrasekhar A, Padrós-Valls R, Pallarès-López R, Palanques-Tost E, Houstis N, Sundt TM, Lee HS, Sodini CG, Aguirre AD. Tissue perfusion pressure enables continuous hemodynamic evaluation and risk prediction in the intensive care unit. Nat Med. 2023 Aug;29(8):1998-2006. doi: 10.1038/s41591-023-02474-6. Epub 2023 Aug 7. PMID: 37550417.
```
--------

## Requirements

The code was tested on Python 3 with NumPy `1.21.5`, SciPy `1.7.3`, and Scikit-learn `1.0.2`.

## Code

Two `.py` are provided. The first one, `beat_to_beat.py`, provides a function called `beat_to_beat_features`, which given a blood pressure waveform its timestmaps and sampling frequency, will perform a beat-to-beat analysis to compute the mean (MAP), systolic (SBP), and diastolic blood pressure (DBP), the heart rate (HR), the pulse pressure (PP) and the time associated to every beat. It also provides a helper function to apply a lowpass filter to a signal.

The second file, `tpp_estimation.py`, provides three functions. One to estimate critical closing pressure given a set of paired MAP and PP*Hr measurements (`pcrit_estimation`), a second one, that given the beat-to-beat features of a blood pressure waveform will estimate critical closing pressure and tissue perfusion pressure with a sliding window approach (`sliding_window_pcrit_estimation_from_features`), and a third one that given a blood pressure waveform and it's sampling frequency, will estimate critical closing pressure and tissue perfusion pressure with a sliding window approach (`sliding_window_pcrit_estimation_from_waveform`).

### Functions
```python
lowpass_filter(signal: np.ndarray, order: int, cutoff: int, fs: int)
```
Implementation of a lowpass filter.

**Parameters:**
- `signal` (np.ndarray): Signal to apply the filter on.
- `order` (int): The order of the filter.
- `cutoff` (int): Cutoff (critical) frequency.
- `fs` (int): signal sampling frequency.

**Returns:**
- np.ndarray of the filttered signal.
--------
```python
beat_to_beat_features(art_vals: np.ndarray, art_time: np.ndarray, fs: int = 120, twin: float = 0.1)
```
Function to extract beat-to-beat mean (map), systolic (sbp), and diastolic (dbp) blood pressure, heart rate (hr), and pulse pressure (pp) from a continuous blood pressure waveform.
 
**Parameters:**
- `art_vals` (np.ndarray): NumPy array with the ABP waveform values.
- `art_time` (np.ndarray): NumPy array with the time vector of the ABP measurements.
- `fs` (int): Sampling frequency.
- `twin` (float): Time window in seconds to look for the original minimum.
- `by` (str): Extrema to use to segment cardiac cycles. Either min or max.
- `filt_outliers` (bool): Use an outlier filter. This may slow down the processing time.

**Returns:**
- A dictionary (Dict[str, np.ndarray]) with the detection of the beat-to-beat bp features. The keys of the dictionary are `map`, `sbp`, `dbp`, `pp`, `hr`, and `time`.
--------
```python
pcrit_estimation(maps: np.ndarray, pps_hrs: np.ndarray)
```    
Estimate critical closing pressure given a set of MAP and PP*HR paired data points. To estimate critical closing pressure, the intercept of the cloud of points formed by the MAP vs PP*HR plot is used. That is, an estimation of the pressure when there is 0 flow. To remove outliers, just points between the 5 and 95 percentil of PP*HR are used.

**Parameters:**
- `maps` (np.ndarray): Set of MAP data points.
- `pps_hrs` (np.ndarray): Set of PP*HR data points.

**Returns:**
- A tuple (Tuple[Float, Float, Float, np.ndarray])with the estimated critical closing pressure, the slope and the r2 square of the linear fit, and a NumPy array with the indices of the data points used for the estimation.
--------
```python
sliding_window_pcrit_estimation_from_features(maps: np.ndarray, pps: np.ndarray, hrs: np.ndarray, time: np.ndarray, window: int = 1, step: int = 1)
```     
Estimate critical closing pressure for a continuous set of beat-to-beat values of mean arterial pressure, pulse pressure, and heart rate using a sliding window approach.

**Parameters:**
- `maps` (np.ndarray0: Value array of MAP values.
- `pps` (np.ndarray): Value array with PP values.
- `hrs` (np.ndarray): Value array with hr values.
- `time` (np.ndarray): Time array in seconds.
- `window` (int): Look-back window size in minutes.
- `step` (int): Sliding window step in minutes.

**Returns:**
- Returns a dictionary (Dict[str, np.ndarray]) with the estimation of critical closing pressure (pcrit), tissue perfusion pressure (tpp), and the average map, pp, hr, pp_hr of the window using a sliding window of size `window` minutes and a step of `step` minutes. It also provides the slope and the r2 of every fit. The keys of the dictionary are pcrit, tpp, slope, r2, map, pp, hr, pp_hr, and time.
--------
```python
sliding_window_pcrit_estimation_from_waveform(art_vals: np.ndarray, art_time: np.ndarray, fs: int = 120, window: int = 1, step: int = 1)
```
Estimate critical closing pressure for a continuous blood pressure waveform using a sliding window approach.

**Parameters:**
- `art_vals` (np.ndarray): NumPy array with the ABP waveform values.
- `art_time` (np.ndarray): NumPy array with the time vector of the ABP measurements.
- `fs` (int): Sampling frequency.
- `window` (int): Look-back window size in minutes.
- `step` (int): Sliding window step in minutes.
    
**Returns:**
- Returns a dictionary (Dict[str, np.ndarray]) with the estimation of critical closing pressure (pcrit), tissue perfusion pressure (tpp), and the average map, pp, hr, pp_hr of the window using a sliding window of size `window` minutes and a step of `step` minutes. It also provides the slope and the r2 of every fit. The keys of the dictionary are pcrit, tpp, slope, r2, map, pp, hr, pp_hr, and time.
