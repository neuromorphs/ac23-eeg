%% Path to the EEGLAB repo
addpath '../../eeglab/plugins/clean_rawdata/'
addpath('../../eeglab');
addpath(genpath('../../eeglab/functions'));
eeg_getversion()

%% Import data
%filename = 'sub-matthias_ses-S001_task-Default_run-001_diverse_bpf_1-50Hz.mat';
%filepath = ['/Volumes/GoogleDrive/.shortcut-targets-by-id/1mvHxk9Ra9K7MmjQxaRYiZKwUw4Y8o841/AC23/DATA/active_piano/sub-matthias/ses-S001/diverse/', ... 
%    filename];
filename = 'sub-matthias_ses-S001_task-Default_run-001_diverse-wrong_bpf_1-50Hz.mat';
filepath = ['/Volumes/GoogleDrive/.shortcut-targets-by-id/1mvHxk9Ra9K7MmjQxaRYiZKwUw4Y8o841/AC23/DATA/active_piano/sub-matthias/ses-S001/diverse-wrong/', ... 
    filename];
load(filepath)

%% Separate stim and EEG channels
eeg = data(1:30, :); % 29 channels + A2
stim = data(end, :);

%% View data and events (sanity check)
eegplot(eeg, 'srate', srate, 'events', event)

%% Standard deviations as a proxy to find bad channels
max(std(eeg(1:29, :)'))

%% Manually drop bad channels from visual inspection
%eeg_chans = 1:30;
%bad_chans = [30, 29, 16]; % from visual inspection on 'diverse'
%chanlocs(bad_chans).labels

eeg_chans = 1:30; % P7 already dropped
bad_chans = [19]; % don't drop any channels 
chanlocs(bad_chans).labels

use_chans = eeg_chans;
for ch = bad_chans
    use_chans = use_chans(use_chans~=ch);
end

%% Drop bad chans
eeg = eeg(use_chans, :);

%% Find longest breaks between events (to be used as baseline periods)
figure(1); clf;
min_break_duration = 30; % seconds
plot(diff([event.latency]) / srate)
stop_event_idx = find(diff([event.latency]) / srate > min_break_duration); 

%% Find corresponding time indices of each baseline segment
calib_idx = [];

for idx = stop_event_idx
    start_idx = event(idx).latency + event(idx).duration;
    stop_idx = event(idx+1).latency; % start of the next event

    % collect time indices of baseline segments
    calib_idx = [calib_idx, start_idx:stop_idx]; % these are discontinuous -- is that a problem?
end

calib_idx = int32(calib_idx); % cast to integer to use for indexing
baseline_data = eeg(:, calib_idx); 

%% Plot baseline data
figure(2); clf;
t = (0:length(eeg)-1)/srate; 
plot(t, eeg, 'k') % what are the units?
hold on
plot(t(calib_idx), baseline_data) % what are the units?
xlabel('Time', 'FontSize',14)
ylabel('Amplitude (uV)', 'FontSize',14)
title('EEG Time Series (Entire Experiment)', 'FontSize', 18)

%% Baseline time series
eegplot(baseline_data, 'srate', srate)

%% Baseline spectra
[spectra,freqs,speccomp,contrib,specstd] = spectopo(...
    baseline_data ...
    , 0 ...
    , srate ...
);
    %, 'limits', limits ... %, 'plotmean', 'on', ...
    %, 'plot', 'off' ...
    %, 'freqfac', osr ...
    %, 'winsize', winsize ...
    %, 'overlap', overlap ...
    % 'title', 'Before ASR', 'reref', 'averef', 

%% Run ASR calibration
%cutoff,blocksize,B,A,window_len
cutoff = 10;
blocksize = [];
B = [];
A = [];
window_len = 0.1; % in seconds, default is 0.5

state = asr_calibrate(...
    baseline_data ...
    , srate ...
    , cutoff, blocksize, B, A, window_len);

%% Apply the processing function to data that shall be cleaned

% Time range to apply ASR filter to
raw_idx = 1:length(eeg); 
raw_idx = setdiff(raw_idx, calib_idx); % exclude calibraion data

raw_data = eeg(:, raw_idx);

% apply the processing to the data (sampling rate 500Hz); 
% see documentation of the function for optional parameters.
clean_data = asr_process(raw_data, srate, state, window_len);

%eegplot(accdata, 'srate', srate)
%fig_acc = gcf;

%% Time series: before and after ASR
figure(3); clf;

t_clean_data = t(raw_idx);

subplot(221);
plot(t_clean_data, raw_data) % what are the units?
title('Before ASR', 'FontSize', 18)
xlabel('time (s)', 'FontSize', 12)
ylabel('Amplitude', 'FontSize', 12)

subplot(222);
plot(t_clean_data, clean_data) % what are the units?
title('After ASR', 'FontSize', 18)
xlabel('time (s)', 'FontSize', 12)
ylabel('Amplitude', 'FontSize', 12)

subplot(223);
[spectra,freqs,speccomp,contrib,specstd] = spectopo(...
    raw_data ...
    , 0 ...
    , srate ...
    , 'plot', 'off'...
);
plot(freqs, spectra)
ylim([-50, 50])
xlabel('Frequency (Hz)', 'FontSize', 12)
ylabel('10log_{10}(µV^2/Hz)', 'FontSize', 12)

subplot(224);
[spectra,freqs,speccomp,contrib,specstd] = spectopo(...
    clean_data ...
    , 0 ...
    , srate ...
    , 'plot', 'off'...
);
plot(freqs, spectra)
ylim([-50, 50])
xlabel('Frequency (Hz)', 'FontSize', 12)
ylabel('10log_{10}(µV^2/Hz)', 'FontSize', 12)

%% Export cleaned EEG data
%save(['/Volumes/GoogleDrive/.shortcut-targets-by-id/1mvHxk9Ra9K7MmjQxaRYiZKwUw4Y8o841/AC23/DATA/active_piano/sub-matthias/ses-S001/diverse/', ... 
%    'after_asr.mat'], "clean_data", "t_clean_data")

save(['/Volumes/GoogleDrive/.shortcut-targets-by-id/1mvHxk9Ra9K7MmjQxaRYiZKwUw4Y8o841/AC23/DATA/active_piano/sub-matthias/ses-S001/diverse-wrong/', ... 
    'after_asr_cutoff10.mat'], "clean_data", "t_clean_data")