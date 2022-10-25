'''
    clean labels function
'''
import re
import numpy as np

def clean_labels(channel_li):
    '''
    This function cleans a list of channels and returns the new channels
    '''

    new_channels = []
    keep_channels = np.ones(len(channel_li), dtype=bool)
    for i in channel_li:
        # standardizes channel names
        M = re.match(r"(\D+)(\d+)", i)

        # account for channels that don't have number e.g. "EKG", "Cz"
        if M is None:
            M = re.match(r"(\D+)", i)
            lead = M.group(1).replace("EEG", "").strip()
            contact = 0
        else:
            lead = M.group(1).replace("EEG", "").strip()
            contact = int(M.group(2))
 
        new_channels.append(f"{lead}{contact:02d}")

    return new_channels
