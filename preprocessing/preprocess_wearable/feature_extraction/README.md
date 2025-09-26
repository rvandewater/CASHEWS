# Feature Extraction

1. If `--align_with_stay` is set, clip signal to align with stay start and end times
2. Divide signal ito segments of length `--minor_segment_m`
3. Extract PPG peaks from each segment, potentially apply synthetic peak interpolation logic if distance between two peaks is abnormally high
4. (Unused) Extract PPG features using the [PyPPG](https://pyppg.readthedocs.io/en/latest/tutorials/PPG_anal.html) library
5. Extract HRV features using the [NeuroKit2](https://neuropsychology.github.io/NeuroKit/functions/hrv.html#sub-domains) library
    1. If `--kubios_fs` then reduce feature set and mimic as many features as possible from the [Kubios HRV Feature Set](https://www.kubios.com/blog/hrv-analysis-methods/)
