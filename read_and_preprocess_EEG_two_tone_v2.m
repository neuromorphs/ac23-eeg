% Read and preprocess the pilot data collected for the two-tone task in its
% second version (i.e. using compiled videos in PsychoPy, rather than
% generating visual stimuli using PsychoPy and attempting to sinc them with
% auditory stimuli). Data were recorded from Abhinav and Aaron on July 8,
% 2023 using the CGX dry electrode EEG system. Data were converted from the
% Brainvision format in which they were saved using EEGLAB.

addpath('/media/Data/Telluride2023/auditory_attention/eeglab2023.0')

data_dir = '/media/Data/Telluride2023/auditory_attention/EEG_two_tone_pilot';

subjects = {'abhinav','aaron'};
sessions = {1,1};

attend_inst = {'up','down'};

% intervals (in seconds) between attention switch instructions
interval_labels = {'60s','20s','10s','2s','random','60s','20s','10s','2s','random'};
intervals = {60*ones(1,4), 20*ones(1,6), 10*ones(1,12), 2*ones(1,60), [18.1 7.4 2.2 15.6 7.8 4.8 4.1 2.65 18.85 19.1 11.55 4.7 3.15]};

trigger_delay = 0; %delay in seconds from trigger to stimulus onset (should match

for subject_num = 1:numel(subjects)
    subject = subjects{subject_num};
    sub_sessions = sessions{subject_num};

    % indices of triggers used for specific parts of stimulus--check that all
    % match PsychoPy
    if strcmpi(subject,'aaron')
        triggers_60s = [1:2, 49:50];
        triggers_20s = [3:5, 51:53];
        triggers_10s = [6:11, 54:59];
        triggers_2s = [12:41, 60:89];
        triggers_random = [42:48, 90:95];
    elseif strcmpi(subject,'abhinav')
        triggers_60s = [1:2, 49:50];
        triggers_20s = [3:5, 51:53];
        triggers_10s = [6:11, 54:55];
        triggers_2s = 12:41;
        triggers_random = 42:48;
    end

    for sesnum = 1:numel(sub_sessions)
        eval(sprintf('EEG = pop_loadset(''%s%d.set'',data_dir);',subject,sub_sessions(sesnum)));
        raw_data_struct = EEG;
        raw_eeg = double(EEG.data);

        channel_info = EEG.chanlocs;

        eeg_data = double(EEG.data);
        srate = double(EEG.srate);
        triggers = raw_eeg(37,:);
        
        attend_switch_times = find(diff(triggers)>1)+1;
        attend_switch_times(triggers(attend_switch_times-1)~=0) = [];
        attend_switch_times(find(diff(attend_switch_times)<srate)+1)=[]; %remove double-triggers
        attend_switch_times = attend_switch_times-(trigger_delay*srate); %triggers delivered at delay after stimulus onset

        % downsample, keep every 25th point
        eeg_ds = downsample(eeg_data',25);    %here we leave the data transposed, i.e. time X channels      
        srate = srate/25;
        attend_switch_times = attend_switch_times/25;

        % high-pass filter
        d = designfilt('highpassiir','FilterOrder',2,'HalfPowerFrequency',0.5,'SampleRate',500,'DesignMethod','butter');
        eeg_data = filtfilt(d, eeg_ds)';    %here, after filtering, we transpose back to channels X time

        % deal with subject-specific trigger issues
        if strcmpi(subject,'abhinav')
            raw_eeg = [nan(size(eeg_data,1),8415) raw_eeg];
            eeg_data = [nan(size(eeg_data,1),336) eeg_data];
            attend_switch_times = [1 attend_switch_times+336];
        elseif strcmpi(subject,'aaron')
            attend_switch_times = attend_switch_times(2:end);
        end

        save(sprintf('%s/%s%d_preprocessed.mat',data_dir,subject,sub_sessions(sesnum)),'raw_eeg','eeg_data','attend_switch_times','srate','channel_info');
        
        for int_ind = 1:numel(intervals)
            interval_list = intervals{int_ind};
            if strcmpi(subject,'aaron')
                if int_ind==1
                    eval(sprintf('up_list = 1:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('down_list = 2:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('eeg_%s_up = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                    eval(sprintf('eeg_%s_down = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                elseif int_ind==2
                    up_list = [1,3:4,6];
                    down_list = [2, 5];
                    eval(sprintf('eeg_%s_up = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                    eval(sprintf('eeg_%s_down = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                else
                    eval(sprintf('up_list = 2:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('down_list = 1:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('eeg_%s_up = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                    eval(sprintf('eeg_%s_down = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                end
            elseif strcmpi(subject,'abhinav')
                if int_ind==1
                    eval(sprintf('up_list = 1:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('down_list = 2:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('eeg_%s_up = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                    eval(sprintf('eeg_%s_down = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                elseif int_ind==2
                    up_list = [1,3:4,6];
                    down_list = [2, 5];
                    eval(sprintf('eeg_%s_up = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                    eval(sprintf('eeg_%s_down = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                else
                    eval(sprintf('up_list = 2:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('down_list = 1:2:numel(triggers_%s);',interval_labels{int_ind}));
                    eval(sprintf('eeg_%s_up = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(up_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                    eval(sprintf('eeg_%s_down = nan(30,round(max(interval_list)*srate),numel(attend_switch_times(triggers_%s(down_list))));',interval_labels{int_ind},interval_labels{int_ind}))
                end
            end
            ll = 0;
            for ili = up_list
                ll = ll+1;
                eval(sprintf('eeg_%s_up(:,1:round(interval_list(ili)*srate),ll) = eeg_data(1:30,round(attend_switch_times(triggers_%s(ili))):round(attend_switch_times(triggers_%s(ili)))+round(interval_list(ili)*srate)-1);',interval_labels{int_ind},interval_labels{int_ind},interval_labels{int_ind}));
            end
            ll = 0;
            for ili = down_list
                ll = ll+1;
                eval(sprintf('eeg_%s_down(:,1:round(interval_list(ili)*srate),ll) = eeg_data(1:30,round(attend_switch_times(triggers_%s(ili))):round(attend_switch_times(triggers_%s(ili)))+round(interval_list(ili)*srate)-1);',interval_labels{int_ind},interval_labels{int_ind},interval_labels{int_ind}));
            end

            save(sprintf('%s/%s%d_preprocessed.mat',data_dir,subject,sub_sessions(sesnum)),'-append',sprintf('eeg_%s_up',interval_labels{int_ind}),sprintf('eeg_%s_down',interval_labels{int_ind}));
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