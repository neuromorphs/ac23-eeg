# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 11:18:02 2022

Toolbox to easily use Boosting in Python Scripts.
Combination of functions taken from Marlies' codes.

@author: u0150098
"""

from eelbrain import *
import numpy as np
import csv

def run_boosting(epochs, stimulus, fs, start_lag=-0.200, end_lag=0.500):
    # Time axis
    tstep = 1. / fs
    n_times = stimulus.shape[0]
    time = UTS(0, tstep, n_times)
    # Load the EEG sensor coordinates
    sensor = Sensor.from_montage('biosemi32')[:30]
    # Create variables
    feature = NDVar(stimulus.squeeze(), (time,), name='stimulus', info={})
    eeg = NDVar(epochs, (time, sensor), name='eeg', info={})
    bvars = {'basis': 0.1, 'test': 1, 'partitions': 4, 'selective_stopping': 1}
    print(f"Vars: start lag is {start_lag}, end lag is {end_lag}, "+
          f"basis is {bvars['basis']}, test is {bvars['test']}, "+
          f"partitions is {bvars['partitions']}, "+
          f"selective stopping is {bvars['selective_stopping']}")
    results = boosting(eeg, feature, start_lag, end_lag, basis=bvars['basis'],
                       test=bvars['test'], partitions=bvars['partitions'],
                       selective_stopping=bvars['selective_stopping'])
    return results

def get_predicted_eeg(stimulus, fs, results):
    print('Getting Prediction Accuracies')
    # get the trf per fold
    trf_per_fold = [partition.h for partition in results.partition_results]
    # get the testing folds
    stimulus_folds = []
    for [start_idx, stop_idx] in results.splits.split_segments:
        time = UTS(float(0), 1 / float(fs),
                   stimulus.squeeze()[start_idx:stop_idx].shape[0])
        stim = NDVar(stimulus.squeeze()[start_idx:stop_idx],
                     dims=(time,), info={}, name='stimulus')
        stimulus_folds.append(stim)
    # get the y predicted for each fold
    peeg_per_fold = [convolve(trf, testStimulus_per_fold)
                      for trf, testStimulus_per_fold in zip(trf_per_fold, stimulus_folds)]
    peeg = [partition.x for partition in peeg_per_fold]
    return peeg


def extract_trf_data(results):
    is_test = [i for i in str(results.partition_result_data).split(',') if 'test' in i][0].split('=')[-1]
    if is_test:
        o = {'corr_folds': results.r.x,
            'trf_folds': results.h.x,
            'times': results.h_time.times}
    elif ~is_test:
        o = {'corr_folds': np.array([p.r.x
                                     for p in results.partition_results]),
            'trf_folds': np.array([p.h.x
                                   for p in results.partition_results]),
            'times': results.h_time.times}
    return o
