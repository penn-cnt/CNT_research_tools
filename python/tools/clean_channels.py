import re
import numpy as np

def clean_channels(channel_li):
    ''' 
    This function cleans a list of channels
    '''

    non_iEEG = ["EKG", "O", "C", "ECG"]
    
    new_channels = []
    keep_channels = np.ones(len(channel_li), dtype=bool)
    for ind, i in enumerate(channel_li):
        # standardizes channel names
        M = re.match(r"(\D+)(\d+)", i)
        lead = M.group(1).replace("EEG", "").strip()
        contact = int(M.group(2))
        # finds non-iEEG channels, make a separate function
        if lead in non_iEEG:
            keep_channels[ind] = 0
        new_channels.append(f"{lead}{contact:02d}")

    
    return new_channels, keep_channels