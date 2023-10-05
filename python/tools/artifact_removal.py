import numpy as np
import pandas as pd
from typing import Union


def _check_disconnection(
    data: np.ndarray, win_inds: np.ndarray, discon: float
) -> np.ndarray:
    """Detects disconnection artifacts based on the discon threshold."""
    return np.sum(np.abs(data[win_inds, :]), axis=0) < discon


def _check_noise(data: np.ndarray, win_inds: np.ndarray, noise: float) -> np.ndarray:
    """Detects noise artifacts based on the noise threshold."""
    return (
        np.sqrt(np.sum(np.power(np.diff(data[win_inds, :], axis=0), 2), axis=0)) > noise
    )


def identify_artifacts(
    data: Union[pd.DataFrame, np.ndarray],
    fs: float,
    discon: float = 1 / 12,
    noise: float = 15000,
    win_size: float = 1,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Identify artifacts in EEG data based on a threshold approach for detecting disconnections and noise.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): The EEG data to process.
            If a DataFrame, columns are assumed to represent channels.
        fs (float): The sampling frequency of the data in Hz.
        discon (float, optional): Disconnection threshold, below which the sum of absolute values
            in a window is considered a disconnection artifact. Defaults to 1/12.
        noise (float, optional): Noise threshold, above which the root mean square of the
            difference in a window is considered a noise artifact. Defaults to 15000.
        win_size (float, optional): Window size in seconds for artifact detection. Defaults to 1.

    Returns:
        Union[pd.DataFrame, np.ndarray]: A binary mask indicating the presence of artifacts.
            Same type as input data.
    """
    if win_size <= 0:
        raise ValueError("Window size must be positive.")

    # Convert window size from seconds to samples
    win_size = int(win_size * fs)
    is_dataframe = isinstance(data, pd.DataFrame)

    if is_dataframe:
        data_array = data.to_numpy()
    else:
        data_array = data

    # Determine the number of windows and maximum index for windowing
    n_wins = np.ceil(data_array.shape[0] / win_size)
    max_inds = int(n_wins * win_size)

    # Create an array of indices for windowing, with NaN padding for incomplete windows
    all_inds = np.arange(
        max_inds, dtype=float
    )  # float data type so it can accept np.nan values
    all_inds[data_array.shape[0] :] = np.nan
    ind_overlap = np.reshape(all_inds, (-1, int(win_size)))

    # Initialize an empty artifact mask with the same shape as data_array
    artifacts = np.empty_like(data_array)
    artifacts = np.isnan(data_array)  # initial artifact detection based on NaN values

    # Loop through each window of data
    for win_inds in ind_overlap:
        win_inds = win_inds[~np.isnan(win_inds)].astype(
            int
        )  # remove NaN values from window indices
        is_disconnected = _check_disconnection(
            data_array, win_inds, discon
        )  # check for disconnection artifacts
        is_noise = _check_noise(
            data_array, win_inds, noise
        )  # check for noise artifacts

        # Update artifact mask based on disconnection and noise detection
        artifacts[win_inds, :] = np.logical_or(
            artifacts[win_inds, :].any(axis=0), np.logical_or(is_disconnected, is_noise)
        )

    if is_dataframe:
        return pd.DataFrame(artifacts, columns=data.columns, index=data.index)
    else:
        return artifacts
