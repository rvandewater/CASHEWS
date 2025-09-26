import time

import numpy as np
import pandas as pd
import pyPPG
import pyPPG.biomarkers as BM
import pyPPG.fiducials as FP
import pyPPG.ppg_sqi as SQI
import scipy
from dotmap import DotMap
from pyPPG.ppg_bm.statistics import get_statistics


def load_pyppg_ppg_object(ppg_signal, sampling_freq, verbose=False):
    signal = DotMap()
    signal.v = ppg_signal
    signal.name = "ppg"
    # signal.start_sig = 0
    # signal.end_sig = -1
    signal.fs = sampling_freq
    signal.ppg = ppg_signal
    signal.vpg = np.gradient(signal.ppg)
    signal.apg = np.gradient(signal.vpg)
    signal.jpg = np.gradient(signal.apg)
    corr_on = ["on", "dn", "dp", "v", "w", "f"]
    correction = pd.DataFrame()
    correction.loc[0, corr_on] = True
    signal.correction = correction
    if verbose:
        print("Created PPG object")
    return pyPPG.PPG(signal, check_ppg_len=True)


def get_pyppg_fiducials(ppg_obj, verbose=False):
    # calculate fiducial points
    #   on - pulse onset,
    #   sp - systolic peak,
    #   dn - dicrotic notch,
    #   dp - diastolic peak,
    #   u,v,w - min and my of first derivative,
    #   a,b,c,d,e,f - min and max points of second derivative
    start_time = time.time()
    fpex = FP.FpCollection(s=ppg_obj)
    fiducials = fpex.get_fiducials(s=ppg_obj)
    if verbose:
        print(f"Extracted fiducial points in {time.time() - start_time} seconds")
    return pyPPG.Fiducials(fp=fiducials)


def extract_pyppg_features(ppg_obj, fiducials_obj, get_stat=True, verbose=False):
    start_time = time.time()
    # initialise the biomarkers package
    bmex = BM.BmCollection(s=ppg_obj, fp=fiducials_obj)

    # Extract biomarkers
    bm_info = bmex.get_biomarkers(get_stat=get_stat)

    bm_defs, bm_vals = bm_info[:2]

    if len(bm_info) == 2:
        return {"defs": bm_defs, "vals": bm_vals}

    bm_stats = bm_info[2]
    if verbose:
        tmp_keys = bm_stats.keys()
        print(f"Extracted biomarkers in {time.time() - start_time} seconds")
        print("Statistics of the biomarkers:")
        for i in tmp_keys:
            print(i, "\n", bm_stats[i])
    # Create a biomarkers class
    # Returns biomarkers per heartbeat
    return {"defs": bm_defs, "vals": bm_vals, "stats": bm_stats}


def get_ppg_sqi(ppg, fs, annotation, get_stat=True, verbose=False):
    start_time = time.time()
    # Todo may be sketchy with template matching
    ppg_sqi = SQI.get_ppgSQI(ppg=ppg, fs=fs, annotation=annotation)
    sqi_df = pd.DataFrame(
        index=pd.Index(list(range(len(ppg_sqi))), dtype="int64", name="Index of pulse"),
        columns=["PPG_SQI"],
        data=ppg_sqi,
    )
    if verbose:
        print(f"Computed SQI in {time.time() - start_time} seconds")

    if get_stat:
        return sqi_df, get_statistics(None, None, {"SQI": {"PPG_SQI": sqi_df}})
    return sqi_df


def get_all_pyppg_features(ppg_signal, sampling_freq, upsample_freq=75, get_stat=True, verbose=False):
    # PyPPG needs at least a sampling frequency of 75 Hz to work properly
    if upsample_freq < 75:
        upsample_freq = 75
    if sampling_freq < upsample_freq:
        expected_samples = round(len(ppg_signal) * upsample_freq / sampling_freq)
        ppg_signal = scipy.signal.resample(ppg_signal, num=expected_samples)
        sampling_freq = upsample_freq

    # Execute PyPPG Pipeline
    ppg_obj = load_pyppg_ppg_object(ppg_signal, sampling_freq)
    fp_obj = get_pyppg_fiducials(ppg_obj, verbose=verbose)
    bm_info = extract_pyppg_features(ppg_obj, fp_obj, get_stat=get_stat, verbose=verbose)
    sqi_info = get_ppg_sqi(ppg_obj.ppg, ppg_obj.fs, annotation=fp_obj.sp, get_stat=get_stat, verbose=verbose)
    if get_stat:
        bm_info["vals"]["sqi"] = sqi_info[0]
        bm_info["stats"]["sqi"] = sqi_info[1]
    else:
        bm_info["vals"]["sqi"] = sqi_info
    return bm_info, (fp_obj.sp, sampling_freq)
