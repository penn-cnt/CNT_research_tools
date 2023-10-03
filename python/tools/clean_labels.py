import re
from typing import List, Optional


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
