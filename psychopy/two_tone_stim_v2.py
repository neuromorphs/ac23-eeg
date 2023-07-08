from psychopy import sound, visual, event, core
import random
import numpy as np
from scipy import signal
import csv

# Participant ID and session input
participant_id = "Sandeep"
session = 'two_tones_stim'

# Load .wav file
wav_no = '../stim/two_tone/two_tone_no_switch.wav'
sound_no = sound.Sound(wav_no)
wav_20s = '../stim/two_tone/two_tone_20s_switch.wav'
sound_20s = sound.Sound(wav_20s)
wav_10s = '../stim/two_tone/two_tone_10s_switch.wav'
sound_10s = sound.Sound(wav_10s)
wav_2s = '../stim/two_tone/two_tone_2s_switch.wav'
sound_2s = sound.Sound(wav_2s)
wav_random = '../stim/two_tone/two_tone_random_switch.wav'
sound_random = sound.Sound(wav_random)

# Set the window size
win = visual.Window(fullscr=True)

'''
mov = visual.MovieStim(
    win, 
    filename=r'/Users/3x10e8/Documents/GitHub/ac23-eeg/stim/two_tone/2s.mp4',
)
'''

mov = visual.MovieStim(
    win,
    r'../stim/two_tone/2s.mp4',
    # path to video file
    size=win.size,
    flipVert=False,
    flipHoriz=False,
    loop=False,
    noAudio=False,
    # volume=0.1,
    autoStart=False
)

# Instructions
instructions = visual.TextStim(win, text='Press SPACE to start the experiment.')
attend_high_instruction = visual.TextStim(win=win, name='attend_high_instruction',
    text='Focus on the high-pitched tones when you see the arrow pointing upward.',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-1.0);
attend_low_instruction = visual.TextStim(win=win, name='attend_low_instruction',
    text='Focus on the low-pitched tones when you see the arrow pointing downward.',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-3.0);
switch_instruction = visual.TextStim(win=win, name='switch_instruction',
    text='When the arrow points up, focus on the high-pitched tone. When the arrow points down, focus on the low-pitched tone.',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-5.0);
arrow_up = visual.ShapeStim(
    win=win, name='arrow_up', vertices=([-0.05,-0.1],[0.05,-0.1],[0.05,0],[0.1,0],[0,0.1],[-0.1,0],[-0.05,0]),
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-6.0, interpolate=True)
arrow_down = visual.ShapeStim(
    win=win, name='arrow_down', vertices=([0.05,0.1],[-0.05,0.1],[-0.05,0],[-0.1,0],[0,-0.1],[0.1,0],[0.05,0]),
    size=(0.5, 0.5),
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-7.0, interpolate=True)

# Instructions to start the experiment
instructions.draw()
win.flip()
# Space bar press to start the experiment
event.waitKeys(keyList=['space'])

if 0:
    #Show instructions
    attend_high_instruction.draw()
    win.flip()
    core.wait(10)

    #Play the sound
    sound_no.play()
    arrow_up.draw()
    win.flip()
    core.wait(60)
    sound_no.stop()

    #Show instructions
    attend_low_instruction.draw()
    win.flip()
    core.wait(10)

    #Play the sound
    sound_no.play()
    arrow_down.draw()
    win.flip()
    core.wait(60)
    sound_no.stop()

    #Show instructions
    switch_instruction.draw()
    win.flip()
    core.wait(10)

    #Switch every 20 seconds
    num_switches = 3
    sound_20s.play()
    for thisSwitch in range(num_switches):
        duration = 20
        if (thisSwitch % 2) == 0:
            arrow_up.draw()
        else:
            arrow_down.draw()
        win.flip()
        core.wait(duration)
    sound_20s.stop()

    #Switch every 10 seconds
    num_switches = 6
    sound_10s.play()
    for thisSwitch in range(num_switches):
        duration = 10
        if (thisSwitch % 2) == 0:
            arrow_down.draw()
        else:
            arrow_up.draw()
        win.flip()
        core.wait(duration)
    sound_10s.stop()

    # Switch every 2 seconds
    num_switches = 30
    sound_2s.play()
    for thisSwitch in range(num_switches):
        duration = 2
        if (thisSwitch % 2) == 0:
            arrow_down.draw()
        else:
            arrow_up.draw()
        win.flip()
        core.wait(duration)
    sound_2s.stop()

    #durations drawn from exponential distribution in MATLAB
    #switchnum_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    num_switches = 9
    # CHECK THAT THESE TIMES MATCH THE ONES IN save_twotone_audio.m
    rand_durations = (14.2233, 3.4572, 2.4988, 3.4741, 6.3717, 4.8263, 9.7305, 2.3583, 13.0599)#13.3300) #I shaved off a little time from the last block to make it exactly 60 s.
    sound_random.play()
    for thisSwitch in range(num_switches):
        duration = rand_durations[thisSwitch]
        if (thisSwitch % 2) == 0:
            arrow_down.draw()
        else:
            arrow_up.draw()
        win.flip()
        core.wait(duration)
    sound_random.stop()

    #Show instructions
    attend_high_instruction.draw()
    win.flip()
    core.wait(10)

    #Play the sound
    sound_no.play()
    arrow_up.draw()
    win.flip()
    core.wait(60)
    sound_no.stop()

    #Show instructions
    attend_low_instruction.draw()
    win.flip()
    core.wait(10)

    #Play the sound
    sound_no.play()
    arrow_down.draw()
    win.flip()
    core.wait(60)
    sound_no.stop()

    #Show instructions
    switch_instruction.draw()
    win.flip()
    core.wait(10)

    #Switch every 20 seconds
    num_switches = 3
    sound_20s.play()
    for thisSwitch in range(num_switches):
        duration = 20
        if (thisSwitch % 2) == 0:
            arrow_up.draw()
        else:
            arrow_down.draw()
        win.flip()
        core.wait(duration)
    sound_20s.stop()

    #Switch every 10 seconds
    num_switches = 6
    sound_10s.play()
    for thisSwitch in range(num_switches):
        duration = 10
        if (thisSwitch % 2) == 0:
            arrow_down.draw()
        else:
            arrow_up.draw()
        win.flip()
        core.wait(duration)
    sound_10s.stop()


# main loop, exit when the status is finished
mov.play()
while not mov.isFinished:
    # draw the movie
    mov.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov.unload()  # unloads when `mov.status == constants.FINISHED`

# Switch every 2 seconds
num_switches = 30
#sound_2s.play()
for thisSwitch in range(num_switches):
    duration = 2
    if (thisSwitch % 2) == 0:
        arrow_down.draw()
    else:
        arrow_up.draw()
    win.flip()
    
    if thisSwitch == 0:
        sound_2s.play()

    core.wait(duration)
sound_2s.stop()

#durations drawn from exponential distribution in MATLAB
#switchnum_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
num_switches = 9
# CHECK THAT THESE TIMES MATCH THE ONES IN save_twotone_audio.m
rand_durations = (14.2233, 3.4572, 2.4988, 3.4741, 6.3717, 4.8263, 9.7305, 2.3583, 13.0599)#13.3300) #I shaved off a little time from the last block to make it exactly 60 s.
sound_random.play()
for thisSwitch in range(num_switches):
    duration = rand_durations[thisSwitch]
    if (thisSwitch % 2) == 0:
        arrow_down.draw()
    else:
        arrow_up.draw()
    win.flip()
    core.wait(duration)
sound_random.stop()

win.close()
core.quit()