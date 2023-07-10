"""
Analysis of CGX EEG data
DENOISING

# streams contain the data for all streams
# stream 0: impedance of EEG channels
# stream 1: data of EEG channels
# stream 2: glove data
# stream 3: output of event based camera
# stream 4: MIDI stream
# stream 5: audio stream

DSS denoising technique
# Compute original and biased covariance matrices
c0, _ = tscov(data)
data.shape = time*channels*repetition

# In this case the biased covariance is simply the covariance of the mean over
# trials
c1, _ = tscov(bias)
bias.shape = time*channels


# Apply DSS
[todss, _, pwr0, pwr1] = dss.dss0(c0, c1)
z = fold(np.dot(unfold(data), todss), epoch_size=n_samples)

# Find best components
best_comp = np.mean(z[:, 0, :], -1)

"""

import mne
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pyxdf
import numpy as np
import time
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from meegkit import dss
from meegkit.utils import fold, rms, tscov, unfold
import librosa # optional, only needed to convert MIDI keys to notes

# Set your datapath -- now this is for Claire's machine
DATA_PATH = "/Users/clairepelofi/Library/CloudStorage/GoogleDrive-cp2830@nyu.edu/.shortcut-targets-by-id/1vf8-kB4CvShQ8yAy2DIw43NnEtVLNM3z/Telluride 2023 Shared/Topic Areas/AC23/DATA/active_piano/sub-matthias/"
DATA_PATH+= "ses-S001/diverse/sub-matthias_ses-S001_task-Default_run-001_diverse.xdf"

# Read in the LSL streams from the XDF file
START_TIME = time.time()
streams, header = pyxdf.load_xdf(DATA_PATH) # this could take ~80 seconds..
print(f'Loaded in {time.time() - START_TIME} s')

# Get the first time stamp across all streams (read from time_stamps)
first_timestamps = []

for s in streams:  # loop through remaining streams
    s_name = s['info']['name']
    t0 = s['time_stamps'][0]
    print(t0, '\t', s_name)

    first_timestamps.append(t0)

first_timestamp = min(first_timestamps)
print(first_timestamp, '\t', '<== earliest')

lsl_streams = {}  # for collecting time stamps and data

# Identify EEG data and impedance streams separately to collect metadata (channel names, etc)
# Remaining streams are collected in one dict

for s in streams:
    s_name = s['info']['name'][0]
    s_type = s['info']['type'][0]
    print(f'Stream Name: {s_name}\tType: {s_type}')
    print('-' * 50)

    # Get the EEG data stream for CGX
    if ('CGX' in s_name) and (s_type == 'EEG'):
        eeg_data = s['time_series']
        eeg_t = s['time_stamps'] - first_timestamp  # offset first time stamp to t=0
        eeg_ch_names = [ch['label'][0] for ch in s['info']['desc'][0]['channels'][0]['channel']]
        eeg_ch_units = [ch['unit'][0] for ch in s['info']['desc'][0]['channels'][0]['channel']]
        eeg_sfreq = s['info']['effective_srate']
        print(f'Channels: {eeg_ch_names}')
        print(f'Unit: {eeg_ch_units}')
        print(f'Eff. Sampling Rate: {eeg_sfreq} Hz')

    # Get the impedance data stream for CGX
    elif ('CGX' in s_name) and (s_type == 'Impeadance'):  # typo in the stream name?
        z_data = s['time_series']
        z_t = s['time_stamps'] - first_timestamp
        z_ch_names = [ch['label'][0] for ch in s['info']['desc'][0]['channels'][0]['channel']]
        z_ch_units = [ch['unit'][0] for ch in s['info']['desc'][0]['channels'][0]['channel']]
        z_sfreq = s['info']['effective_srate']
        print(f'Channels: {z_ch_names}')
        print(f'Unit: {z_ch_units}')
        print(f'Eff. Sampling Rate: {z_sfreq} Hz')

    # Misc streams
    else:
        lsl_streams[s_type] = {}
        lsl_streams[s_type]['data'] = s['time_series']
        lsl_streams[s_type]['time'] = s['time_stamps'] - first_timestamp
        print('shape:', lsl_streams[s_type]['data'].shape)

    print('=' * 50)

plt.plot(eeg_data)
plt.show()

# preprocess eeg data
# define filters
def bandpass(signal,hp_freq,lp_freq,fs):
    b,a = butter(1,[hp_freq,lp_freq],btype='bandpass',fs=fs)
    signal = lfilter(b,a,signal)
    return signal

hp_freq = 0.5
lp_freq = 40

for ch in range(eeg_data.shape[-1]):
    eeg_data[:, ch] = bandpass(eeg_data[:, ch], hp_freq, lp_freq, 500)

fig = plt.plot(eeg_data)
plt.show()


# downsample both EEG and stream of interest for DSS denoising
fs = 500
new_fs = 100

# which stream of interest? --> glove data
DSS_stream = lsl_streams["imu_data"]["data"]
plt.plot(DSS_stream)
plt.show()

print(f'Downsampling from {fs} Hz to {new_fs} Hz')
new_shape = eeg_data.shape[0]*new_fs/fs
old_shape = eeg_data.shape[0]
eeg_data_rs = mne.filter.resample(eeg_data.astype('float64'), new_shape, old_shape, axis=0)

# align small discrepancy between eeg and glove data
DSS_stream = DSS_stream[0:eeg_data_rs.shape[0],]
eeg_data_rs = eeg_data_rs[:,1:]
DSS_stream_rep = np.tile(DSS_stream, (1, 6))

# Conduct DSS
# Compute original and biased covariance matrices
c0, _ = tscov(eeg_data_rs)
plt.imshow(c0, cmap=None, interpolation=None)
plt.colorbar()
plt.title("Covariance Matrix - EEG data")
plt.show()


# In this case the biased covariance is simply the covariance of the mean over
# trials
c1, _ = tscov(DSS_stream_rep)
plt.imshow(c1, cmap=None, interpolation=None)
plt.colorbar()
plt.title("Covariance Matrix - Noise channel")
plt.show()

# Apply DSS
todss, _, pwr0, pwr1 = dss.dss0(c0, c1)
z = fold(np.dot(unfold(eeg_data_rs), todss), epoch_size=eeg_data_rs.shape[0])

# Find best components
best_comp = np.mean(z[:, 0, :], -1)

plt.plot(best_comp)
plt.show()

