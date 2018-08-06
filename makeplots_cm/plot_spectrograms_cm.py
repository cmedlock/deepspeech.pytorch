import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('..')

from python_speech_features import sigproc
from data.data_loader_cm import load_audio

import params_cm

# Read in parameters
sample_rate_ = params_cm.sample_rate
window_size_ = params_cm.window_size
window_stride_ = params_cm.window_stride
window_ = params_cm.window

wav_dir_ = '/data/LibriSpeech_dataset/train_clean_100/wav/'
spect_dir_ = '/data/LibriSpeech_dataset/train_clean_100/spect/'
plot_dir_ = '/data/LibriSpeech_dataset/train_clean_100/spect_plots/'

fig = plt.figure(figsize=(8,4))

counter = 0
n_files = len(os.listdir(wav_dir_))
print('# of files = ',n_files)
for fname in ['1116-132847-0018.wav']:#os.listdir(wav_dir_)[:5]:
    # Load data
    audio_path = wav_dir_+fname
    speech = load_audio(audio_path)

    # Compute spectrogram
    frame_len_ = sample_rate_*window_size_
    frame_step_ = sample_rate_*window_stride_
    frames = sigproc.framesig(speech,frame_len=frame_len_,frame_step=frame_step_)
    n_frames = frames.shape[0]
    ndft = frames.shape[1]
    spect = 1/np.sqrt(ndft)*np.fft.fft(frames,ndft)
    spect = spect.transpose()

    # Only positive frequencies
    spect = spect[:int(ndft/2+1),:] # each column is the DFT of a single frame
    spect = spect[::-1,:] # low frequencies at bottom of axis
    spect = np.abs(spect)
    
    n_frames = spect.shape[1]
    freqs = np.arange(spect.shape[0])/ndft*16.
    
    fig.clear()
    ax = fig.add_subplot(111)
    ax.imshow(spect,aspect='auto',cmap='binary',extent=[0,n_frames*10/1000,freqs[0],freqs[-1]])
    ax.set_xlabel('Time (s)',fontsize=12)
    ax.set_ylabel('Frequency (kHz)',fontsize=12)
    fig.savefig(plot_dir_+fname[:-4]+'.png')

    counter += 1
    
print('Done')
