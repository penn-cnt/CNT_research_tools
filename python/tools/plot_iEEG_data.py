import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Tuple, Union


def plot_iEEG_data(
    data: Union[pd.DataFrame, np.ndarray],
    fs: float,
    start_time_usec: float,
    stop_time_usec: float,
    title: str = "iEEG Data",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots iEEG data, with each channel offset vertically for better visibility.
    NaN values are replaced with zeros for visualization purposes.

    Parameters:
        data (Union[pd.DataFrame, np.ndarray]): The iEEG data to plot. If a DataFrame,
            columns are assumed to represent channels.
        fs (float): The sampling frequency of the data in Hz.
        start_time_usec (float): The start time in microseconds.
        stop_time_usec (float): The stop time in microseconds.
        title (str, optional): The title of the plot. Defaults to 'iEEG Data'.

    Returns:
        Tuple[plt.Figure, plt.Axes]: The figure and axis objects.
    """

    # Replace NaN values with 0 and reverse channel order for visualization purposes
    if isinstance(data, pd.DataFrame):
        data = data.fillna(0)
        data = data.iloc[:, ::-1]
    elif isinstance(data, np.ndarray):
        data = np.nan_to_num(data)
        data = data.iloc[:, ::-1]

    # Convert start and stop times to seconds
    start_time_sec = start_time_usec / 1e6
    stop_time_sec = stop_time_usec / 1e6

    # Convert start and stop times to indices
    start_idx = int(start_time_sec * fs)
    stop_idx = int(stop_time_sec * fs)

    # Slice data and create a new time vector based on the selected duration
    data = data[start_idx:stop_idx]
    t_sec = np.linspace(start_time_sec, stop_time_sec, num=data.shape[0])

    # Create a figure and a single set of axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate vertical offset for each channel
    offsets = (
        np.arange(data.shape[1]) * 200
    )  # Adjust the multiplier to control vertical spacing

    # Loop through each channel and plot the data with vertical offset
    for i in range(data.shape[1]):
        ax.plot(
            t_sec, data.iloc[:, i] + offsets[i], color="black", linewidth=0.8
        )  # Set color to black and decrease linewidth

    # Hide the spines (borders)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove y-axis ticks and labels
    ax.set_yticks([])
    # Adjust y-limits to reduce the gap between x-axis and first channel
    ax.set_ylim(
        [offsets[0] - 200, offsets[-1] + 200]
    )  # Adjust y-limits to include the first and last channel with some space

    # Set the x-axis limit to start at start_time_sec
    ax.set_xlim(t_sec[0], t_sec[-1])

    # Set x-label and title
    ax.set_xlabel("Time (s)", fontsize=12)
    fig.suptitle(title, fontsize=14, y=0.95)

    # Add channel labels at the y-axis location of each channel trace
    for i, offset in enumerate(offsets):
        ax.text(
            -0.02,
            offset,
            data.columns[i],
            transform=ax.get_yaxis_transform(),
            ha="right",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to accommodate the title

    return fig, ax
