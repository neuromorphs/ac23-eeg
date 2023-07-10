% Read and preprocess the pilot data collected for the two-tone task in its
% first version (i.e. using PsychoPy to generate visual cues and syncing 
% them with auditory stimuli). In this original version, the auditory 
% stimulus was restarted every time a trigger was sent, resulting in an 
% audible discontinuity. Data were recorded from Sandeep on July 1,2023 
% using the CGX dry electrode EEG system. Data were converted from the
% Brainvision format in which they were saved using EEGLAB.

addpath('/media/Data/Telluride2023/auditory_attention/eeglab2023.0')

data_dir = '/media/Data/Telluride2023/auditory_attention/EEG_two_tone_pilot';

subjects = {'sandeep'};
sessions = {[2 3]};

attend_inst = {'up','down'};

% intervals (in seconds) between attention switch instructions
interval_labels = {'60s','20s','10s','2s','random'};
intervals = {60*ones(1,2), 20*ones(1,3), 10*ones(1,6), 2*ones(1,30), [14.2233, 3.4572, 2.4988, 3.4741, 6.3717, 4.8263, 9.7305, 2.3583, 13.0599]};

% indices of triggers used for specific parts of stimulus--check that all
% match PsychoPy
triggers_60s = 1:2;
triggers_20s = 3:5;
triggers_10s = 6:11;
triggers_2s = 12:41;
triggers_random = 42:50;

trigger_delay = 0.1; %delay in seconds from trigger to stimulus onset (should match

for subject_num = 1:numel(subjects)
    subject = subjects{subject_num};
    sub_sessions = sessions{subject_num};
    for sesnum = 1:numel(sub_sessions)
        eval(sprintf('EEG = pop_loadset(''%s%d.set'',data_dir);',subject,sub_sessions(sesnum)));
        raw_data_struct = EEG;
        raw_eeg = double(EEG.data);

        channel_info = EEG.chanlocs;

        eeg_data = double(EEG.data);
        srate = double(EEG.srate);
        triggers = raw_eeg(37,:);
        
        attend_switch_times = find(diff(triggers)>1)+1;
        attend_switch_times = attend_switch_times-(trigger_delay*srate); %triggers delivered at delay after stimulus onset
        
        % downsample, keep every 25th point
        eeg_ds = downsample(eeg_data',25);    %here we leave the data transposed, i.e. time X channels      
        srate = srate/25;
        attend_switch_times = attend_switch_times/25;

        % high-pass filter
        d = designfilt('highpassiir','FilterOrder',2,'HalfPowerFrequency',0.5,'SampleRate',500,'DesignMethod','butter');
        eeg_data = filtfilt(d, eeg_ds)';    %here, after filtering, we transpose back to channels X time

        save(sprintf('%s/%s%d_lightly_preprocessed.mat',data_dir,subject,sub_sessions(sesnum)),'raw_eeg','eeg_data','attend_switch_times','srate','channel_info');
        
        for int_ind = 1:numel(intervals)
            interval_list = intervals{int_ind};
            if int_ind<3
                eval(sprintf('up_list = 1:2:numel(triggers_%s);',interval_labels{int_ind}));
                eval(sprintf('down_list = 2:2:numel(triggers_%s);',interval_labels{int_ind}));
                eval(sprintf('eeg_%s_up = nan(size(eeg_data,1),round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                eval(sprintf('eeg_%s_down = nan(size(eeg_data,1),round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
            else
                eval(sprintf('up_list = 2:2:numel(triggers_%s);',interval_labels{int_ind}));
                eval(sprintf('down_list = 1:2:numel(triggers_%s);',interval_labels{int_ind}));
                eval(sprintf('eeg_%s_up = nan(size(eeg_data,1),round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                eval(sprintf('eeg_%s_down = nan(size(eeg_data,1),round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
            end
            ll = 0;
            for ili = up_list
                ll = ll+1;
                eval(sprintf('eeg_%s_up(:,1:round(interval_list(ili)*srate),ll) = eeg_data(:,round(attend_switch_times(triggers_%s(ili))):round(attend_switch_times(triggers_%s(ili)))+round(interval_list(ili)*srate)-1);',interval_labels{int_ind},interval_labels{int_ind},interval_labels{int_ind}));
            end
            ll = 0;
            for ili = down_list
                ll = ll+1;
                eval(sprintf('eeg_%s_down(:,1:round(interval_list(ili)*srate),ll) = eeg_data(:,round(attend_switch_times(triggers_%s(ili))):round(attend_switch_times(triggers_%s(ili)))+round(interval_list(ili)*srate)-1);',interval_labels{int_ind},interval_labels{int_ind},interval_labels{int_ind}));
            end

            save(sprintf('%s/%s%d_lightly_preprocessed.mat',data_dir,subject,sub_sessions(sesnum)),'-append',sprintf('eeg_%s_up',interval_labels{int_ind}),sprintf('eeg_%s_down',interval_labels{int_ind}));
        end
    end
end


% load(sprintf('%s/%s%d.mat',data_dir,subject,sub_sessions(sesnum)));
% triggers = [];
% for ind = 1:nume(urevent)
%     if mod(ind,2)==0
%         continue
%     else
%         triggers = [triggers urevent(ind).latency];
%     end
% end