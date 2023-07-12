'''
This example uses mne-realtime, and is based on the examples here:
https://mne.tools/mne-realtime/auto_examples/plot_compute_rt_average.html#sphx-glr-auto-examples-plot-compute-rt-average-py
https://mne.tools/mne-realtime/auto_examples/plot_lslclient_rt.html

Requirements:
    Install mne
        pip3 install mne
    Install mne-realtime
        pip3 install https://api.github.com/repos/mne-tools/mne-realtime/zipball/main
'''

import sys
sys.path.append('./mne-realtime/')
from mne_realtime import LSLClient, RtEpochs
import mne
import pylsl
import numpy as np
import time
import matplotlib.pyplot as plt
from blitting import BlitManager # https://matplotlib.org/stable/tutorials/advanced/blitting.html


# first resolve an EEG stream on the lab network
print("Looking for all LSL streams...")
streams = pylsl.resolve_stream()
print("="*50)

source_id = [] # placeholder for EEG stream's source_id
for info in streams:

    # Get relevant params from CGX's EEG stream:
    if (info.type().lower() == 'eeg') and ('cgx' in info.name().lower()):
        source_id = info.source_id()
        channel_count = info.channel_count()
        nominal_srate = info.nominal_srate()
        print('-'*50)
        print('CGX EEG stream identified with')
        print('source_id:', source_id)
        print('channel_count:', channel_count)
        print('nominal_srate:', nominal_srate)
        print('-'*50)

    else: # print out other streams in case we need something
        print(
            info.type(), 
            info.name(), 
            info.channel_count(),
            info.nominal_srate(), 
            info.source_id(), 
            #info.hostname(),
            #info.uid(),
            )

if source_id == []: # no EEG stream was identified!
    print('ERROR: no CGX EEG stream found. Check if LabStreamingLayer was started...')

# Provide channel names for the headset used
ch_names = []

# Corresponding mne channel types
ch_types = []

if channel_count == 27: # Channel map for CGX Quick20r (from acquistion software)
    n_eeg_chans = 19
    ch_names.extend(['F7', 'Fp1', 'Fp2', 'F8', 'F3', 'Fz', 'F4', 'C3'])
    ch_names.extend(['Cz', 'P8', 'P7', 'Pz', 'P4', 'T3', 'P3', 'O1'])
    ch_names.extend(['O2', 'C4', 'T4'])

    ch_types.extend(['eeg']*n_eeg_chans)

elif channel_count == 37: # Channel map for CGX Quick32r (from acquistion software)
    n_eeg_chans = 29 # 29 eeg channels
    ch_names.extend(['AF7', 'Fpz', 'F7', 'Fz', 'T7', 'FC6', 'Fp1', 'F4', 'C4', 'Oz'])
    ch_names.extend(['CP6', 'Cz', 'PO8', 'CP5', 'O2', 'O1', 'P3', 'P4', 'P7', 'P8'])
    ch_names.extend(['Pz', 'PO7', 'T8', 'C3', 'Fp2', 'F3', 'F8', 'FC5', 'AF8'])
    
    ch_types.extend(['eeg']*n_eeg_chans)
    #ch_types.extend(['eeg']) # in case A2 is used

# extra ground
ch_names.extend(['A2'])
# aux electrodes
ch_names.extend(['ExG 1', 'ExG 2'])
# accelerometer
ch_names.extend(['ACC22', 'ACC23', 'ACC24'])
# packet count
ch_names.extend(['Packet Count'])
# trigger
ch_names.extend(['trigger'])

ch_types.extend(['misc']) # in case A2 is used
ch_types.extend(['emg']*2)
ch_types.extend(['misc']*3) # for ACC channels
ch_types.extend(['misc']) # for packet count
ch_types.extend(['stim']) # for trigger

# Create an info object to use with the LSL client below
info = mne.create_info(
    ch_names = ch_names,
    sfreq = nominal_srate,
    ch_types = ch_types,
)
info.set_montage('standard_1020', match_case=False)

# Make an LSL client using the info and CGX stream's source_id (as host)
rt_client = LSLClient(
    info = info,
    host = source_id,
    # wait_max = ,
    buffer_size = 1000*5,
)

print(rt_client.get_measurement_info())

rt_client.start()

n_samples = 500 * 4
wait_time = 4

if 0:
    samples, _ = rt_client.client.pull_chunk(
        max_samples = n_samples,
        timeout = wait_time,
    )
    data = np.vstack(samples).T
    data[:n_eeg_chans, :]/=1e6 
    print(data)
    print(data.shape)


# make a new figure
fig, ax = plt.subplots(2, 1)
plt.xlabel('Time')


# this is the max wait time in seconds until client connection
wait_max = 10

# For this example, let's use the mock LSL stream.
n_epochs = 10
picks = ['eeg'] # ['O2'] # can only use EEG channels here?
if picks == ['eeg']:
    chan_idx_to_plot = range(n_eeg_chans)
else:
    chan_idx_to_plot = range(len(picks))

# let's observe ten seconds of data
for ii in range(n_epochs):

    print('Got epoch %d/%d' % (ii + 1, n_epochs))
    epoch = rt_client.get_data_as_epoch(
        n_samples=int(nominal_srate), 
        picks = picks,
    )
    epoch_avg_data = epoch.average().get_data() # is this necessary?
    #mne.viz.plot_topomap(epoch, pos=info, axes=ax)

    if ii == 0:
        # add a line
        #print(epoch.average().plot(axes=ax))
        #epoch.average().plot(axes=ax[1])

        #im = mne.viz.plot_topomap(epoch_avg_data[:, 0])
        #ax[1].imshow(im, animated=True,)

        lns = []
        for ch in chan_idx_to_plot:
            ln, = ax[0].plot(
                epoch_avg_data[ch, :], 
                animated=True,
                linewidth=.5,
                )
            lns.append(ln)
        #ln1, = ax.plot(epoch_avg_data[1, :], animated=True)
        #print(ln)
        plt.show(block=False)

        bm = BlitManager(fig.canvas, lns)
        # make sure our window is on the screen and drawn
        #plt.pause(.1)

    else:
        # update the artists
        for ch in chan_idx_to_plot:
            lns[ch].set_ydata(epoch_avg_data[ch, :])

        # tell the blitting manager to do its thing
        bm.update()


print('Streams closed')