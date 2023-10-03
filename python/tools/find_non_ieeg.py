from typing import List
import re
import numpy as np

non_ieeg = ["EKG", "O", "C", "ECG"]


def find_non_ieeg(channel_li: List[str]) -> np.ndarray:
    """
    Identifies non-iEEG channel labels from a given list of channel labels.

    Parameters:
    - channel_li (List[str]): A list of channel labels as strings.

    Returns:
    - np.ndarray: A boolean numpy array where each element corresponds to whether
                  the respective channel label in the input list is a non-iEEG channel.
    """

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
