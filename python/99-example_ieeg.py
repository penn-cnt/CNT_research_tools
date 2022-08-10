#%%
# pylint: disable-msg=C0103
# pylint: disable-msg=C0301
#%%
# Imports
import matplotlib.pyplot as plt
import numpy as np

import tools

# %%
username = "pattnaik"
password = 'pat_ieeglogin.bin'

iEEG_filename = "HUP172_phaseII"
start_time_usec = 402580 * 1e6
stop_time_usec = 402800 * 1e6
electrodes = ["LE10","LE11","LH01","LH02","LH03","LH04"]

# %%
# data, fs = tools.get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, select_electrodes=electrodes)
data, fs = tools.get_iEEG_data(username, password, iEEG_filename, start_time_usec, stop_time_usec)

# %%
display(data.columns)

# %%
clean_channels, keep_channels = tools.clean_channels(data.columns)

# %%
data.columns = clean_channels
data = data.iloc[:, keep_channels]

#%%
display(data.columns)

# %% Plot the data
t_sec = np.linspace(start_time_usec, stop_time_usec, num=data.shape[0]) / 1e6
fig, ax = tools.plot_iEEG_data(data, t_sec)
fig.set_size_inches(18.5, 10.5)
ax.set_title(iEEG_filename)
fig.show()

# %%
