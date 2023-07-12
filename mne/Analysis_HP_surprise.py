# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 08:43:05 2022
Analyze Violin Data - TRF on envelope predicting video eeg data
@author: cpelofi
"""

# %%
import eelbrain
import numpy as np
import pyxdf
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch, filtfilt
from scipy.io.wavfile import read, write
import mne
from boosting_toolbox import run_boosting, extract_trf_data
from scipy.signal import hilbert, chirp
from scipy.io import loadmat


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

fpath = "/Users/clairepelofi/Library/CloudStorage/GoogleDrive-cp2830@nyu.edu/.shortcut-targets-by-id/1vf8-kB4CvShQ8yAy2DIw43NnEtVLNM3z/Telluride 2023 Shared/Topic Areas/AC23/DATA/HarryPotter Multilingual SingleSpeaker/eeg recordings/07-07-22/Claire/sub-Claire_ses-S001_task-Default_run-001_calibration.xdf"
streams, header = pyxdf.load_xdf(filename) # this could take ~80 seconds..



flist = [f for f in os.listdir(fpath) if 'vo' in f]
# %%
epochs = []
stimulus = []
# %%
hp_freq = 1
lp_freq = 25
# %%


# %% Figure params
matplotlib.use('Qt5Agg')
plt.interactive(True)
# %%


streams, header = pyxdf.load_xdf(fpath+'/'+fname)
        # %%
eeg_stream = [s for s in streams if 'CGX' in s['info']['name'][0] and 'Impedance' not in s['info']['name'][0]][0]
trig_stream = [s for s in streams if 'violin' in s['info']['name'][0]][0]
# %%
eeg_data = eeg_stream['time_series'][:,:30]
eeg_t = eeg_stream['time_stamps']

# %% EEG preprocess
for ch in range(eeg_data.shape[-1]):
    # subtract A2 from eeg
    #eeg_data[:,ch] -= eeg_data[:,:].mean(-1)
        eeg_data[:,ch] = notch_filter(eeg_data[:,ch], 500, [60])
        eeg_data[:, ch] = mne.filter.filter_data(eeg_data[:, ch].astype('float64'), 500, 0.5, 40)
        #eeg_data[:,ch] = bandpass(eeg_data[:,ch],hp_freq,lp_freq, 500)
        eeg_data[:,ch] = normalize_data(eeg_data[:,ch])
    eeg_t -= 121e-3 # measured using plot
        # %%

        trig_data = trig_stream['time_series']
        trig_t = trig_stream['time_stamps']
        if trig_data.shape[0] < 2:
            # %% reconstruct trigger
            if trig_data[0,0] == 1:
                trig_data = np.array([trig_data[0,0],2])
                trig_t = np.array([trig_t[0],trig_t[0]+audio_duration])
            elif trig_data[0,0] == 2:
                trig_data = np.array([1,trig_data[0,0]])
                trig_t = np.array([trig_t[0]-audio_duration,trig_t[0]])

        # %% Envelope
        new_fs = 500
        if env_tech == 'Kyle':
            name_env = wav_file
            name_env = name_env.split(".")[0] + ".npy"
            env = np.load(wav_path + name_env)

            # change sr here
            new_shape = env.shape[0] * new_fs / fs_audio
            old_shape = env.shape[0]

            # env = mne.filter.resample(env.astype('float64'), new_shape, old_shape, axis=0)
            # env = bandpass(env, hp_freq=1, lp_freq=25, fs=500)
            env = mne.filter.filter_data(env, 48000, None, 40)
            print(f'Downsampling {wav_file} from {fs_audio} Hz to {new_fs} Hz')
            env = mne.filter.resample(env.astype('float64'), new_shape, old_shape, axis=0)
            env = mne.filter.filter_data(env, 500, 0.5, None)


        elif env_tech == 'Hilbert':
            name_env = wav_file
            name_env = name_env.split(".")[0] + ".mat"
            env = loadmat(wav_path+name_env)
            env = env["env"].squeeze()
        env_t = np.linspace(0, env.shape[0] / new_fs, env.shape[0]) + trig_t[0]

        # %%
        wait_secs = 2
        # %%
        start_stamp_eeg = np.abs(eeg_t - (trig_t[0] + wait_secs)).argmin()
        stop_stamp_eeg = np.abs(eeg_t - (trig_t[-1] - wait_secs)).argmin()
        # %%
        start_stamp_env = np.abs(env_t - (trig_t[0] + wait_secs)).argmin()
        stop_stamp_env = np.abs(env_t - (trig_t[-1] - wait_secs)).argmin()

        # %%
        eeg_data = eeg_data[start_stamp_eeg:stop_stamp_eeg, :]
        env = env[start_stamp_env:stop_stamp_env]

        # %% plot signals together
        # plt.figure()
        # plt.plot(eeg_data)
        # plt.plot(env)
        # plt.show()

        # Fix length of eeg and sound
        if eeg_data.shape[0] > env.shape[0]:
            print('eeg longer than env')
            eeg_data = eeg_data[:env.shape[0], :]
        else:
            print('env longer than eeg')
            env = env[:eeg_data.shape[0]]

        # epochs.append(eeg_data[:env.shape[0]])
        epochs.append(eeg_data)
        stimulus.append(env)
# %%

epochs_stack = np.vstack(epochs)
stimulus_stack = np.hstack(stimulus)
# %%
boosting_results = dict()
fs = 500
new_fs = 128

print(f'Downsampling from {fs} Hz to {new_fs} Hz')
new_shape = epochs_stack.shape[0]*new_fs/fs
old_shape = epochs_stack.shape[0]
stimulus_rs = mne.filter.resample(stimulus_stack.astype('float64'), new_shape, old_shape, axis=0)
epochs_rs = mne.filter.resample(epochs_stack.astype('float64'), new_shape, old_shape, axis=0)

print('Running Boosting')
boosting_results = run_boosting(epochs_rs, stimulus_rs, new_fs)
print('---')

# %%
trf_data = boosting_results.h_scaled.x
corr = boosting_results.r.x
print(corr.max())

t = np.linspace(-200, 500, trf_data.shape[-1])
channels = [a['label'] for a in eeg_stream['info']['desc'][0]['channels'][0]['channel']][:30]

# %% Figure
fig, (ax1, ax2) = plt.subplots(2)
fig.set_size_inches(7, 7)
fig.suptitle('P001: VO Conditions Average TRF')

ax1.plot(t, trf_data.T,label=None)
ax1.plot(t, trf_data.mean(0),'k--')
ax1.set(xlabel='Time (ms)', ylabel='Amplitude (AU)')
ax1.legend(channels)
ax2.hist(corr)
ax2.set(xlabel='R-value', ylabel='# electrode')
#plt.show()


'''
# predict data with envelope
eeg_pred = eelbrain.convolve(epochs_rs, boosting_results.h_scaled)
corr_pred_vid =  eelbrain.correlation_coefficient(eeg_pred, eeg_video_rs)
corr_pred_vid2 =  eelbrain.cross_correlation(eeg_pred, eeg_video_rs)

'''
