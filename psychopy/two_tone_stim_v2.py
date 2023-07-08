from psychopy import prefs
prefs.hardware['audiolib'] = ['pyo']
from psychopy import sound, visual, event, core, clock
import random
import numpy as np
from scipy import signal
import csv

# Participant ID and session input
participant_id = "Tim"
session = 'two_tones_stim'

# Path for stimulus files
#STIM_PATH = r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\'

# Load .wav file
#wav_no = '../stim/two_tone/two_tone_no_switch.wav'
#sound_no = sound.Sound(wav_no)
#wav_20s = '../stim/two_tone/two_tone_20s_switch.wav'
#sound_20s = sound.Sound(wav_20s)
#wav_10s = '../stim/two_tone/two_tone_10s_switch.wav'
#sound_10s = sound.Sound(wav_10s)
#wav_2s = '../stim/two_tone/two_tone_2s_switch.wav'
#sound_2s = sound.Sound(wav_2s)
#wav_random = '../stim/two_tone/two_tone_random1_switch.wav'
#sound_random = sound.Sound(wav_random)

# Set the window size
win = visual.Window(fullscr=True)

'''
mov = visual.MovieStim(
    win, 
    filename=r'/Users/3x10e8/Documents/GitHub/ac23-eeg/stim/two_tone/2s.mp4',
)
'''

mov_60sup = visual.MovieStim(
    win,
    '', #r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\up_video.mp4',
    # path to video file
    size=win.size,
    flipVert=False,
    flipHoriz=False,
    loop=False,
    noAudio=False,
    # volume=0.1,
    autoStart=False
)

if 1:
    mov_60sdown = visual.MovieStim(
        win,
        '',
        # path to video file
        size=win.size,
        flipVert=False,
        flipHoriz=False,
        loop=False,
        noAudio=False,
        # volume=0.1,
        autoStart=False
    )

    mov_20s = visual.MovieStim(
        win,
        '',
        # path to video file
        size=win.size,
        flipVert=False,
        flipHoriz=False,
        loop=False,
        noAudio=False,
        # volume=0.1,
        autoStart=False
    )

    mov_10s = visual.MovieStim(
        win,
        '',
        # path to video file
        size=win.size,
        flipVert=False,
        flipHoriz=False,
        loop=False,
        noAudio=False,
        # volume=0.1,
        autoStart=False
    )

    mov_2s = visual.MovieStim(
        win,
        '',
        # path to video file
        size=win.size,
        flipVert=False,
        flipHoriz=False,
        loop=False,
        noAudio=False,
        # volume=0.1,
        autoStart=False
    )

    mov_random1 = visual.MovieStim(
        win,
        '',
        # path to video file
        size=win.size,
        flipVert=False,
        flipHoriz=False,
        loop=False,
        noAudio=False,
        # volume=0.1,
        autoStart=False
    )

mov_random2 = visual.MovieStim(
    win,
    '',
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
#arrow_up = visual.ShapeStim(
#    win=win, name='arrow_up', vertices=([-0.05,-0.1],[0.05,-0.1],[0.05,0],[0.1,0],[0,0.1],[-0.1,0],[-0.05,0]),
#    size=(0.5, 0.5),
#    ori=0.0, pos=(0, 0), anchor='center',
#    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
#    opacity=None, depth=-6.0, interpolate=True)
#arrow_down = visual.ShapeStim(
#    win=win, name='arrow_down', vertices=([0.05,0.1],[-0.05,0.1],[-0.05,0],[-0.1,0],[0,-0.1],[0.1,0],[0.05,0]),
#    size=(0.5, 0.5),
#    ori=0.0, pos=(0, 0), anchor='center',
#    lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
#    opacity=None, depth=-7.0, interpolate=True)

# Instructions to start the experiment
instructions.draw()
win.flip()
# Space bar press to start the experiment
event.waitKeys(keyList=['space'])

#Show instructions
attend_high_instruction.draw()
win.flip()
core.wait(10)

# attend to high, slow tone sequence for 60 s
ISI_60sup = clock.StaticPeriod()
ISI_60sup.start(1)
mov_60sup.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\up_video.mp4')
ISI_60sup.complete()
mov_60sup.play()
while not mov_60sup.isFinished:
    # draw the movie
    mov_60sup.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov_60sup.unload()  # unloads when `mov.status == constants.FINISHED`

if 1:
    #Show instructions
    attend_low_instruction.draw()
    win.flip()
    core.wait(10)

    # attend to low, fast tone sequence for 60 s
    ISI_60sdown = clock.StaticPeriod()
    ISI_60sdown.start(1)
    mov_60sdown.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\down_video.mp4')
    ISI_60sdown.complete()
    mov_60sdown.play()
    while not mov_60sdown.isFinished:
        # draw the movie
        mov_60sdown.draw()
        # flip buffers so they appear on the window
        win.flip()

    # stop the movie, this frees resources too
    mov_60sdown.unload()  # unloads when `mov.status == constants.FINISHED`

    #Show instructions
    switch_instruction.draw()
    win.flip()
    core.wait(10)

    #Switch every 20 seconds
    ISI_20s = clock.StaticPeriod()
    ISI_20s.start(1)
    mov_20s.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\20s_video.mp4')
    ISI_20s.complete()
    mov_20s.play()
    while not mov_20s.isFinished:
        # draw the movie
        mov_20s.draw()
        # flip buffers so they appear on the window
        win.flip()

    # stop the movie, this frees resources too
    mov_20s.unload()  # unloads when `mov.status == constants.FINISHED`
    
    #Show instructions
    switch_instruction.draw()
    win.flip()
    core.wait(10)

    #Switch every 10 seconds
    ISI_10s = clock.StaticPeriod()
    ISI_10s.start(1)
    mov_10s.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\10s_video.mp4')
    ISI_10s.complete()
    mov_10s.play()
    while not mov_10s.isFinished:
        # draw the movie
        mov_10s.draw()
        # flip buffers so they appear on the window
        win.flip()

    # stop the movie, this frees resources too
    mov_10s.unload()  # unloads when `mov.status == constants.FINISHED`
    
    #Show instructions
    switch_instruction.draw()
    win.flip()
    core.wait(10)

    # Switch every 2 seconds
    ISI_2s = clock.StaticPeriod()
    ISI_2s.start(1)
    mov_2s.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\2s_video.mp4')
    ISI_2s.complete()
    mov_2s.play()
    while not mov_2s.isFinished:
        # draw the movie
        mov_2s.draw()
        # flip buffers so they appear on the window
        win.flip()

    # stop the movie, this frees resources too
    mov_2s.unload()  # unloads when `mov.status == constants.FINISHED`
    
    #Show instructions
    switch_instruction.draw()
    win.flip()
    core.wait(10)

    #Switch randomly
    ISI_random1 = clock.StaticPeriod()
    ISI_random1.start(1)
    mov_random1.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\random1_video.mp4')
    ISI_random1.complete()
    mov_random1.play()
    while not mov_random1.isFinished:
        # draw the movie
        mov_random1.draw()
        # flip buffers so they appear on the window
        win.flip()

    # stop the movie, this frees resources too
    mov_random1.unload()  # unloads when `mov.status == constants.FINISHED`


#Show instructions
attend_high_instruction.draw()
win.flip()
core.wait(10)

# attend to high, slow tone sequence for 60 s
ISI_60sup = clock.StaticPeriod()
ISI_60sup.start(1)
mov_60sup.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\up_video.mp4')
ISI_60sup.complete()
mov_60sup.play()
while not mov_60sup.isFinished:
    # draw the movie
    mov_60sup.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov_60sup.unload()  # unloads when `mov.status == constants.FINISHED`

#Show instructions
attend_low_instruction.draw()
win.flip()
core.wait(10)

# attend to low, fast tone sequence for 60 s
ISI_60sdown = clock.StaticPeriod()
ISI_60sdown.start(1)
mov_60sdown.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\down_video.mp4')
ISI_60sdown.complete()
mov_60sdown.play()
while not mov_60sdown.isFinished:
    # draw the movie
    mov_60sdown.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov_60sdown.unload()  # unloads when `mov.status == constants.FINISHED`

#Show instructions
switch_instruction.draw()
win.flip()
core.wait(10)

#Switch every 20 seconds
ISI_20s = clock.StaticPeriod()
ISI_20s.start(1)
mov_20s.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\20s_video.mp4')
ISI_20s.complete()
mov_20s.play()
while not mov_20s.isFinished:
    # draw the movie
    mov_20s.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov_20s.unload()  # unloads when `mov.status == constants.FINISHED`

#Show instructions
switch_instruction.draw()
win.flip()
core.wait(10)

#Switch every 10 seconds
ISI_10s = clock.StaticPeriod()
ISI_10s.start(1)
mov_10s.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\10s_video.mp4')
ISI_10s.complete()
mov_10s.play()
while not mov_10s.isFinished:
    # draw the movie
    mov_10s.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov_10s.unload()  # unloads when `mov.status == constants.FINISHED`

#Show instructions
switch_instruction.draw()
win.flip()
core.wait(10)

# Switch every 2 seconds
ISI_2s = clock.StaticPeriod()
ISI_2s.start(1)
mov_2s.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\2s_video.mp4')
ISI_2s.complete()
mov_2s.play()
while not mov_2s.isFinished:
    # draw the movie
    mov_2s.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov_2s.unload()  # unloads when `mov.status == constants.FINISHED`

#Show instructions
switch_instruction.draw()
win.flip()
core.wait(10)

#Switch randomly
ISI_random2 = clock.StaticPeriod()
ISI_random2.start(1)
mov_random2.load(r'C:\Users\Andreou LabAdmin\Documents\PsychoPy\stim\two_tone\random2_video.mp4')
ISI_random2.complete()
mov_random2.play()
while not mov_random2.isFinished:
    # draw the movie
    mov_random2.draw()
    # flip buffers so they appear on the window
    win.flip()

# stop the movie, this frees resources too
mov_random2.unload()  # unloads when `mov.status == constants.FINISHED`

win.close()
core.quit()