import logging

import neurokit2 as nk
import numpy as np
import pandas as pd


def calculate_kubios_features(hrv_features, peaks, sampling_freq):
    # Time-Domain Features
    # Using SDANN1 instead of SDANN5, as the minor segments are already 5min time windows
    drop_time_features = [
        f"HRV_{feature}"
        for feature in [
            "SDANN5",
            "SDANN2",
            "SDNNI5",
            "SDNNI2",
            "SDSD",
            "CVNN",
            "CVSD",
            "MedianNN",
            "MadNN",
            "MCVNN",
            "IQRNN",
            "SDRMSSD",
            "Prc20NN",
            "Prc80NN",
        ]
    ]
    hrv_features.drop(columns=drop_time_features, inplace=True, errors="ignore")

    # Frequency-Domain Features
    hrv_features.rename({"HRV_LFn": "HRV_LFr", "HRV_HFn": "HRV_HFr"}, inplace=True)
    # Add additional features
    if "HRV_LF" in hrv_features.columns and "HRV_TP" in hrv_features.columns:
        hrv_features["HRV_LFn"] = hrv_features["HRV_LF"] / hrv_features["HRV_TP"]
    # Drop features
    hrv_features.drop(columns=["HRV_ULF", "HRV_VHF", "HRV_TP"], errors="ignore", inplace=True)

    # Nonlinear Features
    drop_nonlinear_features = [
        f"HRV_{feature}"
        for feature in [
            "S",
            "CSI",
            "CVI",
            "CSI_Modified",
            "GI",
            "SI",
            "AI",
            "PI",
            "SD1d",
            "SD1a",
            "C1d",
            "C1a",
            "SD2d",
            "SD2a",
            "C2d",
            "C2a",
            "SDNNd",
            "SDNNa",
            "Cd",
            "Ca",
            "PIP",
            "IALS",
            "PSS",
            "PAS",
            "FuzzyEn",
            "CMSEn",
            "RCMNEs",
            "HFD",
            "KFD",
            "LZC",
        ]
    ]
    hrv_features.drop(columns=drop_nonlinear_features, errors="ignore", inplace=True)
    hrv_features.drop(
        columns=[col for col in hrv_features.columns if col.startswith("HRV_MFDFA")],
        errors="ignore",
        inplace=True,
    )
    try:
        rqa_features = nk.hrv_rqa(peaks, sampling_freq)
        hrv_features["HRV_REC"] = rqa_features["RecurrenceRate"]
        hrv_features["HRV_DET"] = rqa_features["Determinism"]
    except Exception as e:
        logging.getLogger(__name__).warning(f"RQA features could not be extracted: {e}")
        pass

    return hrv_features


def get_all_hrv_features(
    sampling_freq,
    peaks=None,
    ppg_signal=None,
    psd_method="welch",
    kubios_features_only=False,
    show=False,
    verbose=False,
):
    # psd_method can be fft or welch
    assert peaks is not None or ppg_signal is not None, "Either peaks or ppg_signal must be provided"
    if peaks is None:
        try:
            peaks, info = nk.ppg_peaks(ppg_signal, sampling_freq, correct_artifacts=True)
        except Exception as e:
            if verbose:
                logging.info(f"Error identifying peaks, skipping HRV features. {e}")
            return None
        peaks = peaks["PPG_Peaks"]
        peak_indices = info["PPG_Peaks"]
    elif np.mean(peaks) > 1:
        peak_indices = peaks
        # expects list of zeros with ones at peak indices
        padding = int(np.mean(np.diff(peaks)) // 2)
        new_peaks = np.zeros((np.max(peaks) + padding + 1,), dtype=np.int64)
        new_peaks[peaks] = 1
        peaks = new_peaks
    else:
        peak_indices = np.argwhere(peaks).squeeze()

    if len(peak_indices) < 2:
        if verbose:
            logging.info("Not enough peaks for HRV features.")
        return None

    try:
        hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_freq, show=show)
    except Exception as e:
        hrv_time = pd.DataFrame()
        logging.getLogger(__name__).warning(f"HRV features could not be extracted: {e}")

    try:
        hrv_freq = nk.hrv_frequency(
            peaks,
            sampling_rate=sampling_freq,
            psd_method=psd_method,
            show=show,
            vlf=((0, 0.04) if kubios_features_only else (0.0033, 0.04)),
            interpolation_rate=(4 if kubios_features_only else 100),
        )
    except Exception as e:
        hrv_freq = pd.DataFrame()
        logging.getLogger(__name__).warning(f"HRV frequency features could not be extracted: {e}")

    try:
        hrv_nl = nk.hrv_nonlinear(
            peaks,
            sampling_rate=sampling_freq,
            show=show,
            interpolation_rate=(4 if kubios_features_only else 100),
        )
    except Exception as e:
        hrv_nl = pd.DataFrame()
        logging.getLogger(__name__).warning(f"HRV nonlinear features could not be extracted: {e}")
    if "HRV_SampEn" in hrv_nl.columns:
        hrv_nl["HRV_SampEn"] = hrv_nl["HRV_SampEn"].replace([np.inf, -np.inf], np.nan)
    if "HRV_DFA_alpha2" not in hrv_nl.columns:
        # For some windows, the signal data is too short to compute this feature, we need to add it manually for compatibility
        hrv_nl["HRV_DFA_alpha2"] = np.nan

    hrv_features = pd.concat([hrv_time, hrv_freq, hrv_nl], axis=1)

    if kubios_features_only:
        hrv_features = calculate_kubios_features(hrv_features, peaks, sampling_freq)

    return hrv_features
