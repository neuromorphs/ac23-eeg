# -*- coding: utf-8 -*-

# TODO: look at other brain data, do we have more? Maybe record tomorrow? From Claire and Mahmoud? someone who doe snot speak french
# TODO: check permutation method, could be better?
# TODO: comapre with envelope, does that predict better?

import sys
sys.path.append("/Users/clairepelofi/Dropbox/Mac/Documents/GitHub/ac23-eeg/mne")

import numpy as np
import pyxdf
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch, filtfilt
import mne
from boosting_toolbox2 import run_boosting, extract_trf_data
from numpy.random import permutation
import matplotlib
matplotlib.use("Qt5Agg")


def highpass(signal, hp_freq):
    b,a = butter(1, hp_freq, btype='high', fs=500)
    signal = filtfilt(b,a,signal)
    return signal

def lowpass(signal, lp_freq):
    b,a = butter(1, lp_freq, btype='low', fs=fs_audio)
    signal = lfilter(b,a,signal)
    return signal

def bandpass(signal,hp_freq,lp_freq,fs):
    b,a = butter(1,[hp_freq,lp_freq],btype='bandpass',fs=fs)
    signal = lfilter(b,a,signal)
    return signal

def normalize_data(x):
    x = x/x.std()
    x = x-x.mean()
    return x

def notch_filter(y, fs, notch_filter_frequencies):
    for f0 in notch_filter_frequencies:
        Q = 30.0  # Quality factor
        b, a = iirnotch(f0, Q, fs)  # Design notch filter
        y = lfilter(b, a, y, axis=0)
    return y

# %%
PATH = "/Users/clairepelofi/Library/CloudStorage/GoogleDrive-cp2830@nyu.edu/.shortcut-targets-by-id/1vf8-kB4CvShQ8yAy2DIw43NnEtVLNM3z/Telluride 2023 Shared/Topic Areas/AC23/DATA/HarryPotter Multilingual SingleSpeaker"
date = "/eeg recordings/07-07-22"
subj = "/Claire/"
data_file = [f for f in os.listdir(PATH+date+subj) if 'hp.xdf' in f][0]
language = "french"

print('************')
print(subj)
print(language)

filename = PATH + date + subj + data_file
print('Loading xdf file...')
streams, header = pyxdf.load_xdf(filename) # this could take ~80 seconds..

epochs = []
stimulus = []
# %%
hp_freq = 1
lp_freq = 25
fs = 500

# %%
eeg_stream = [s for s in streams if 'CGX' in s['info']['name'][0] and 'Impedance' not in s['info']['name'][0]][0]
eeg_data = eeg_stream['time_series'][:,:30]
eeg_t = eeg_stream['time_stamps']

for ch in range(eeg_data.shape[-1]):
    eeg_data[:,ch] = notch_filter(eeg_data[:,ch], fs, [60])
    eeg_data[:,ch] = bandpass(eeg_data[:,ch],hp_freq,lp_freq,fs)
    eeg_data[:,ch] = normalize_data(eeg_data[:,ch])


# plot components explaining variance
# plt.figure(30)
# plt.clf()
# plt.plot(eeg_t, eeg_data)
# plt.show(block=True)

start_trigger = []
stop_trigger = []

# deal with triggers for Claire
# claire_start_trigger = [104.9170902*fs, 783.8683153*fs, 1634.5745120000001*fs]
# claire_start_trigger = [int(np.ceil(value)) for value in claire_start_trigger]
# start_trigger = np.array(claire_start_trigger)
#
# claire_stop_trigger = [729.5414412*fs, 1511.6913976*fs, 2272.7242122*fs]
# claire_stop_trigger = [int(np.ceil(value)) for value in claire_stop_trigger]
# stop_trigger = np.array(claire_stop_trigger)


# deal with triggers for Kyle
kyle_start_trigger = [155.2602066*fs, 915.2142538*fs, 1799.5053372999998*fs]
kyle_start_trigger = [int(np.ceil(value)) for value in kyle_start_trigger]
start_trigger = np.array(kyle_start_trigger)

kyle_stop_trigger = [779.905467*fs, 1643.0416695*fs, 2437.8192609000002*fs]
kyle_stop_trigger = [int(np.ceil(value)) for value in kyle_stop_trigger]
stop_trigger = np.array(kyle_stop_trigger)

# trials IDs
tab_path = PATH + date + subj
tab_file = [f for f in os.listdir(tab_path) if '.tab' in f][0]

with open(tab_path+tab_file) as f:
    text = f.readlines()
trial_id = []
for line in text:
    if 'C:' in line and 'trial_id' in line:
        trial_id.append(line.split('\\')[-1].split('.')[0][2:])

epochs = {}
for i, [start_time, stop_time] in enumerate(zip(start_trigger, stop_trigger)):
    print(start_time, stop_time)
    epochs[trial_id[i]] = (eeg_data[start_time:stop_time, :])

new_fs = 100

print(f'Downsampling from {fs} Hz to {new_fs} Hz')
new_shape = epochs[language].shape[0]*new_fs/fs
old_shape = epochs[language].shape[0]
eeg_data_rs = mne.filter.resample(epochs[language].astype('float64'), new_shape, old_shape, axis=0)

# load surprise
surp_file = PATH + "/surps/" + f'HP{language}_surps_100Hz.txt'
print(f'HP{language}_surps_100Hz.txt')
surprise = np.loadtxt(surp_file)

# Realign EEG and Surprise if needed
if len(surprise) > len(eeg_data_rs):
    print("Adjusting Surprise length")
    surprise = surprise[0:len(eeg_data_rs)]
elif len(surprise) < len(eeg_data_rs):
    print("Adjusting EEG Data length")
    eeg_data_rs = eeg_data_rs[0:len(surprise)]


# %% TRF ANALYSIS
boosting_results = dict()
print('Running Boosting')
boosting_results = run_boosting(eeg_data_rs, surprise, new_fs)
print('---')

trf_data = boosting_results.h_scaled.x
corr = boosting_results.r.x
print(corr.max())

t = np.linspace(-200, 500, trf_data.shape[-1])
channels = [a['label'] for a in eeg_stream['info']['desc'][0]['channels'][0]['channel']][:30]

# %% Figure
plt.figure(10)
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(7, 7)
fig.suptitle('Real model TRF and suprise HP')

ax1.plot(t, trf_data.T,label=None)
ax1.plot(t, trf_data.mean(0),'k--')
ax1.set(xlabel='Time (ms)', ylabel='Amplitude (AU)')
ax1.legend(channels)
ax2.hist(corr)
ax2.set(xlabel='R-value', ylabel='# electrode')
plt.show(block=True)

# %% Perform the permutation test
permut_num = 100  # should be 100
NULL_TRF = np.empty((0,) + trf_data.shape)
NULL_CORR = np.empty((0,) + corr.shape)

for repetition in range(permut_num):
    print(f"Iteration {repetition + 1}:")
    print('---')

    nz_index = np.nonzero(surprise)[0]
    nz_val = surprise[nz_index]
    np.random.shuffle(nz_val)
    shuf_surprise = np.copy(surprise)
    shuf_surprise[nz_index] = nz_val

    null_boosting_results = dict()

    print('Running Boosting')
    null_boosting_results = run_boosting(eeg_data_rs, shuf_surprise, new_fs)
    print('---')

    null_trf_data = null_boosting_results.h_scaled.x
    null_corr = null_boosting_results.r.x
    print(null_corr.max())
    t = np.linspace(-200, 500, null_trf_data.shape[-1])

    NULL_TRF = np.append(NULL_TRF, np.expand_dims(null_trf_data, axis=0), axis=0)
    NULL_CORR = np.append(NULL_CORR, np.expand_dims(null_corr, axis=0), axis=0)

all_NULL_TRF = np.mean(NULL_TRF,0)
all_NULL_CORR = np.mean(NULL_CORR,0)

plt.figure(20)
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(7, 7)
fig.suptitle('Real model TRF and suprise HP')

ax1.plot(t, all_NULL_TRF.T,label=None)
ax1.plot(t, all_NULL_TRF.mean(0),'k--')
ax1.set(xlabel='Time (ms)', ylabel='Amplitude (AU)')
ax1.legend(channels)
ax2.hist(all_NULL_CORR)
ax2.set(xlabel='R-value', ylabel='# electrode')
# plt.show(block=True)

# Violin plots Corr and null corr
data_violin = [corr, all_NULL_CORR]

import seaborn as sns
import pandas as pd

data = np.concatenate([corr, all_NULL_CORR])
labels = np.concatenate([np.repeat('Real model', len(corr)), np.repeat('Null model', len(all_NULL_CORR))])
df = pd.DataFrame({'Vectors': labels, 'Values': data})

# Plot the violin plot using Seaborn
plt.figure(30)
sns.violinplot(x='Vectors', y='Values', data=df)
plt.xlabel('Vectors')
plt.ylabel('Values')
plt.title('Compare ' + f'{language} model - not native')
plt.show(block=True)

from scipy import stats
t_statistic, p_value = stats.ttest_rel(corr, all_NULL_CORR)

# Print the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)