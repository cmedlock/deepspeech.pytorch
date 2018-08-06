import torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('..')

import librosa
from librosa import display
from python_speech_features import sigproc
from data.data_loader_cm import load_audio

import params_cm

# Read in parameters
sample_rate_ = params_cm.sample_rate
window_size_ = params_cm.window_size
window_stride_ = params_cm.window_stride
window_ = params_cm.window

wav_dir_ = '/data/LibriSpeech_dataset/train_clean_100/wav/'
plot_dir_ = '/data/LibriSpeech_dataset/train_clean_100/mfcc_plots/'

fig = plt.figure(figsize=(8,4))

counter = 0
n_files = len(os.listdir(wav_dir_))
print('# of files = ',n_files)
for fname in ['1116-132847-0018.wav']:#os.listdir(wav_dir_)[:5]:
    # Load data
    audio_path = wav_dir_+fname
    speech = load_audio(audio_path)

    # Compute Mel spectrogram
    frame_len_ = sample_rate_*window_size_
    frame_step_ = sample_rate_*window_stride_
    mfcc = librosa.feature.melspectrogram(speech,sr=sample_rate_,n_fft=int(frame_len_),\
                                          hop_length=int(frame_step_),power=1,n_mels=26)
    
    img = librosa.display.specshow(mfcc,sr=sample_rate_,hop_length=int(frame_step_),\
                                   x_axis='time',y_axis='mel',fmax=sample_rate_/2,cmap='binary')
    
    plt.savefig(plot_dir_+fname[:-4]+'.png')
    
print('Done')
