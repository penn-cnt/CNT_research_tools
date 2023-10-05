import os
import re
import time
import pickle
import logging
import numpy as np
import pandas as pd
import logging.handlers
from typing import List, Union, Optional, Tuple
from ieeg.auth import Session

enable_logging = True
if enable_logging:
    log_filename = os.path.join(os.pardir, "logs", "iEEG_data_retrieval.log")
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for more detailed logs
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_filename, maxBytes=5 * 1024 * 1024, backupCount=3
            ),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger()
    logger.info("Logger initialized.")
else:
    logger = logging.getLogger()
    logger.disabled = True

MAX_RETRIES = 50
SLEEP_DURATION = 1
CLIP_DURATION_LIMIT = int(120e6)
MAX_CHANNELS_AT_ONCE = 100
CLIP_SIZE = int(60e6)
CHANNEL_SIZE = 20


def _pull_iEEG(
    dataset: object, start_usec: int, duration_usec: int, channel_ids: List[int]
) -> Union[np.array, None]:
    """
    Retrieve data from an iEEG dataset within a specified time range and for specified channels.

    Parameters:
    - dataset (object): The iEEG dataset object.
    - start_usec (int): The start time in microseconds.
    - duration_usec (int): Duration for which data is to be pulled in microseconds.
    - channel_ids (list): List of channel IDs to pull the data from.

    Returns:
    - array or None: Returns the data as an array if successful, otherwise None.
    """
    for _ in range(MAX_RETRIES):
        try:
            return dataset.get_data(start_usec, duration_usec, channel_ids)
        except Exception:  # TODO: catch specific exceptions
            time.sleep(SLEEP_DURATION)

    logger.error(
        f"Failed to pull data for {dataset.name}, {start_usec / 1e6}, {duration_usec / 1e6}, {len(channel_ids)} channels"
    )
    return None


def get_channel_ids(
    all_channel_labels: List[str],
    electrodes: List[Union[str, int]],
    mode: str = "select",
) -> List[int]:
    """
    Derive channel IDs based on provided labels or IDs and mode (select or ignore).

    Parameters:
    - all_channel_labels (list): List of all available channel labels in the dataset.
    - electrodes (list): List of channel labels or IDs to be selected or ignored.
    - mode (str): Either "select" or "ignore" to determine behavior. Defaults to "select".

    Returns:
    - list: List of channel IDs.
    """
    if isinstance(electrodes[0], int):
        ids = (
            electrodes
            if mode == "select"
            else [i for i in range(len(all_channel_labels)) if i not in electrodes]
        )
    elif isinstance(electrodes[0], str):
        if any([e not in all_channel_labels for e in electrodes]):
            raise ValueError("Channel not in iEEG")
        ids = [
            i
            for i, e in enumerate(all_channel_labels)
            if (e in electrodes) == (mode == "select")
        ]
    else:
        raise TypeError("Electrodes not given as a list of ints or strings")

    return ids


def get_iEEG_data(
    username: str,
    password_bin_file: str,
    iEEG_filename: str,
    start_time_usec: float,
    stop_time_usec: float,
    select_electrodes: Optional[List[Union[str, int]]] = None,
    ignore_electrodes: Optional[List[Union[str, int]]] = None,
    clean_channel_labels: bool = False,
    remove_substr: Optional[str] = None,
    delimiter: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Union[Tuple[pd.DataFrame, float], None]:
    """
    Retrieve iEEG data from a dataset based on specified parameters.

    Parameters:
    - username (str): Username for accessing the dataset.
    - password_bin_file (str): iEEG password bin file for accessing the dataset.
    - iEEG_filename (str): The filename or ID of the iEEG dataset.
    - start_time_usec (float): The start time in microseconds.
    - stop_time_usec (float): The stop time in microseconds.
    - select_electrodes (list, optional): List of channel labels or IDs to be specifically selected. Defaults to None.
    - ignore_electrodes (list, optional): List of channel labels or IDs to be ignored. Defaults to None.
    - clean_channel_labels (bool, optional): Whether to clean the channel labels using the clean_labels function. Defaults to False.
    - remove_substr (str, optional): A substring to remove from each label when cleaning. Defaults to None.
    - delimiter (str, optional): A delimiter to split and rejoin label parts when cleaning. Defaults to None.
    - output_file (str, optional): If provided, the resulting data will be saved to this file. Defaults to None.

    Returns:
    - tuple or None: If not saving to an output file, returns a tuple of (DataFrame, sample rate). If any error occurs, or if saving to an output file, returns None.
    """
    duration = stop_time_usec - start_time_usec

    # Connecting to a session and open dataset
    for _ in range(MAX_RETRIES):
        try:
            pwd = open(password_bin_file, "r").read()
            session = Session(username, pwd)
            dataset = session.open_dataset(iEEG_filename)
            all_channel_labels = dataset.get_channel_labels()
            break
        except Exception:  # TODO: catch specific exceptions
            time.sleep(SLEEP_DURATION)
    else:
        raise ValueError("Failed to open dataset")
    logger.info(f"Connected to dataset: {iEEG_filename}")

    if clean_channel_labels:
        # Clean all channel labels initially
        all_channel_labels = clean_labels(all_channel_labels, remove_substr, delimiter)
        if select_electrodes:
            select_electrodes = clean_labels(
                select_electrodes, remove_substr, delimiter
            )
        if ignore_electrodes:
            ignore_electrodes = clean_labels(
                ignore_electrodes, remove_substr, delimiter
            )
        logger.info(
            f"Cleaned channel labels with substr: {remove_substr} and delimiter: {delimiter}"
        )

    # Determine channel IDs and names
    if select_electrodes:
        channel_ids = get_channel_ids(
            all_channel_labels, select_electrodes, mode="select"
        )
    elif ignore_electrodes:
        channel_ids = get_channel_ids(
            all_channel_labels, ignore_electrodes, mode="ignore"
        )
    else:
        channel_ids = list(range(len(all_channel_labels)))

    channel_names = [all_channel_labels[i] for i in channel_ids]

    # Fetch data
    data = []
    if duration < CLIP_DURATION_LIMIT and len(channel_ids) < MAX_CHANNELS_AT_ONCE:
        data.append(_pull_iEEG(dataset, start_time_usec, duration, channel_ids))
    else:
        # Determine if breaking by time or by channel
        if duration > CLIP_DURATION_LIMIT:
            for clip_start in range(
                int(start_time_usec), int(stop_time_usec), CLIP_SIZE
            ):
                clip_duration = min(CLIP_SIZE, stop_time_usec - clip_start)
                data.append(_pull_iEEG(dataset, clip_start, clip_duration, channel_ids))
        else:
            for start_idx in range(0, len(channel_ids), CHANNEL_SIZE):
                end_idx = start_idx + CHANNEL_SIZE
                ids_slice = channel_ids[start_idx:end_idx]
                data.append(_pull_iEEG(dataset, start_time_usec, duration, ids_slice))

        # Check if any data fetching failed
        if any(d is None for d in data):
            logger.error("Failed to fetch some data segments.")
            return None

    # Combine data segments
    data = np.concatenate(data, axis=1 if duration <= CLIP_DURATION_LIMIT else 0)

    df = pd.DataFrame(data, columns=channel_names)
    fs = dataset.get_time_series_details(
        dataset.ch_labels[0]
    ).sample_rate  # get sample rate

    logger.info(
        f"Data fetched successfully for {iEEG_filename} from {start_time_usec / 1e6} to {stop_time_usec / 1e6} seconds."
    )

    if output_file:
        try:
            with open(output_file, "wb") as f:
                pickle.dump([df, fs], f)
            logger.info(f"Data successfully written to {output_file}.")
        except Exception as e:
            logger.error(f"Failed to write data to {output_file}. Error: {e}")
    else:
        return df, fs


def clean_labels(
    channel_list: List[str],
    remove_substr: Optional[str] = None,
    delimiter: Optional[str] = None,
) -> List[str]:
    """
    Cleans a list of channel labels by standardizing their format.

    Parameters:
    - channel_list (list): A list of channel labels as strings.
    - remove_substr (str, optional): A substring to remove from each label. Defaults to None.
    - delimiter (str, optional): A delimiter to split and rejoin label parts. Defaults to None.

    Returns:
    - list: A list of cleaned channel labels.
    """
    cleaned_channels = []

    for label in channel_list:
        if remove_substr:
            label = label.replace(remove_substr, "")

        if delimiter:
            label_parts = label.split(delimiter)
            label = delimiter.join(label_parts[1:])  # Skip the first part

        regex_match = re.match(r"(\D+)(\d+)", label)

        if regex_match:
            lead = regex_match.group(1).strip()
            contact = int(regex_match.group(2))
            cleaned_channels.append(f"{lead}{contact:02d}")
        else:
            # If the regex match fails, keep the original label
            cleaned_channels.append(label)

    return cleaned_channels


def find_non_ieeg(channel_li: List[str]) -> np.ndarray:
    """
    Identifies non-iEEG channel labels from a given list of channel labels.

    Parameters:
    - channel_li (List[str]): A list of channel labels as strings.

    Returns:
    - np.ndarray: A boolean numpy array where each element corresponds to whether
                  the respective channel label in the input list is a non-iEEG channel.
    """
    non_ieeg = ["EKG", "O", "C", "ECG"]

    is_non_ieeg = np.zeros(len(channel_li), dtype=bool)
    for ind, i in enumerate(channel_li):
        # Attempt to split channel label into a non-digit part and a digit part
        regex_match = re.match(r"(\D+)(\d+)", i)
        if regex_match is None:
            # If no match is found, skip to the next iteration
            continue

        lead = regex_match.group(1)

        # Check if the non-digit part matches any of the known non-iEEG label prefixes
        if lead in non_ieeg:
            is_non_ieeg[ind] = True

    return is_non_ieeg
