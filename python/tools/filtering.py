import numpy as np
import pandas as pd
from scipy.signal import iirnotch, sosfiltfilt, butter, filtfilt, lfilter
from typing import Optional, Union


def lowpass_filter(
    data: Union[np.ndarray, pd.DataFrame],
    fs: float,
    cutoff: float,
    order: int = 3,
    causal: Optional[bool] = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """Apply a lowpass filter to the data.

    Args:
        data (Union[np.ndarray, pd.DataFrame]): The input data to be filtered.
        fs (float): The sampling frequency of the data.
        cutoff (float): The cutoff frequency of the lowpass filter.
        order (int, optional): The order of the filter. Defaults to 3.
        causal (bool, optional): Whether to apply a causal filter. Defaults to False.

    Returns:
        Union[np.ndarray, pd.DataFrame]: The filtered data.
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.to_numpy()
    else:
        data_array = data

    if causal:
        b, a = butter(order, cutoff, output="ba", fs=fs, btype="low")
        data_filt = lfilter(b, a, data_array, axis=0)
    else:
        sos = butter(order, cutoff, output="sos", fs=fs, btype="low")
        data_filt = sosfiltfilt(sos, data_array, axis=0)

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(data_filt, columns=data.columns, index=data.index)
    else:
        return data_filt


def highpass_filter(
    data: Union[np.ndarray, pd.DataFrame],
    fs: float,
    cutoff: float,
    order: int = 3,
    causal: Optional[bool] = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """Apply a highpass filter to the data.

    Args:
        data (Union[np.ndarray, pd.DataFrame]): The input data to be filtered.
        fs (float): The sampling frequency of the data.
        cutoff (float): The cutoff frequency of the highpass filter.
        order (int, optional): The order of the filter. Defaults to 3.
        causal (bool, optional): Whether to apply a causal filter. Defaults to False.

    Returns:
        Union[np.ndarray, pd.DataFrame]: The filtered data.
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.to_numpy()
    else:
        data_array = data

    if causal:
        b, a = butter(order, cutoff, output="ba", fs=fs, btype="high")
        data_filt = lfilter(b, a, data_array, axis=0)
    else:
        sos = butter(order, cutoff, output="sos", fs=fs, btype="high")
        data_filt = sosfiltfilt(sos, data_array, axis=0)

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(data_filt, columns=data.columns, index=data.index)
    else:
        return data_filt


def notch_filter(
    data: Union[pd.DataFrame, np.ndarray],
    fs: float,
    freq: float = 60,
    quality: float = 30,
    causal: Optional[bool] = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """Apply a notch filter to the data to remove noise at a specified frequency.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): The input data to be filtered.
        fs (float): The sampling frequency of the data.
        freq (float, optional): The frequency to be notched out. Defaults to 60.
        quality (float, optional): The quality factor of the notch filter. Defaults to 30.
        causal (bool, optional): Whether to apply a causal filter. Defaults to False.

    Returns:
        Union[pd.DataFrame, np.ndarray]: The filtered data.
    """
    b, a = iirnotch(freq, quality, fs)
    if causal:
        data_filt = lfilter(b, a, data, axis=0)
    else:
        data_filt = filtfilt(b, a, data, axis=0)

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(data_filt, columns=data.columns, index=data.index)
    else:
        return data_filt


def bandpass_filter(
    data: Union[np.ndarray, pd.DataFrame],
    fs: float,
    order: int = 3,
    lo: float = 1,
    hi: float = 120,
    causal: Optional[bool] = False,
) -> Union[np.ndarray, pd.DataFrame]:
    """Apply a bandpass filter to the data within a specified frequency range.

    Args:
        data (Union[np.ndarray, pd.DataFrame]): The input data to be filtered.
        fs (float): The sampling frequency of the data.
        order (int, optional): The order of the filter. Defaults to 3.
        lo (float, optional): The lower bound of the frequency range. Defaults to 1.
        hi (float, optional): The upper bound of the frequency range. Defaults to 120.
        causal (bool, optional): Whether to apply a causal filter. Defaults to False.

    Returns:
        Union[np.ndarray, pd.DataFrame]: The filtered data.
    """
    if isinstance(data, pd.DataFrame):
        data_array = data.to_numpy()
    else:
        data_array = data

    if causal:
        b, a = butter(order, [lo, hi], output="ba", fs=fs, btype="bandpass")
        data_filt = lfilter(b, a, data_array, axis=0)
    else:
        sos = butter(order, [lo, hi], output="sos", fs=fs, btype="bandpass")
        data_filt = sosfiltfilt(sos, data_array, axis=0)

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(data_filt, columns=data.columns, index=data.index)
    else:
        return data_filt
