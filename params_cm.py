###########################################################
# Fixed parameters for speech recognition
###########################################################

# Model and feature type
model_type = 'DeepSpeech' # DeepSpeech, Wav2Letter, or ResNet50
feature_type = 'spectrogram' # spectrogram, mfcc, or logmel

# Audio sampling parameters
sample_rate   = 16000 # Sample rate
window_size   = 0.02 # Window size for spectrogram in seconds
window_stride = 0.01 # Window stride for spectrogram in seconds
window        = 'hamming' # Window type to generate spectrogram

# Audio noise parameters
noise_dir  = None # directory to inject noise
noise_prob = 0.4 # probability of noise being added per sample
noise_min  = 0.0 # minimum noise level to sample from (1.0 means all noise and no original signal)
noise_max  = 0.5 # maximum noise level to sample from (1.0 means all noise and no original signal)

# Dataset and model save location
# Note: for ResNet50 must use pre-aligned transcription (e.g., TIMIT)
labels_path    = './labels.json' #Contains all characters for prediction
train_manifest = './manifest_files/100_files_manifest.csv' #relative path to train manifest
val_manifest = './manifest_files/100_files_manifest.csv' #relative path to val manifest
model_path = 'models/deepspeech_final.pth' # Location to save best validation model

# Model parameters
hidden_size   = 768 # Hidden size of RNNs
hidden_layers = 5 # Number of RNN layers
bias          = True  # Use biases
rnn_type      = 'rnn' #Type of the RNN. rnn|gru|lstm are supported
rnn_act_type  = 'relu' #Type of the activation within RNN. tanh | relu are supported
bidirectional = False # Whether or not RNN is uni- or bi-directional

# Training parameters
epochs          = 2 # Number of training epochs
learning_anneal = 1.1 # Annealing applied to learning rate every epoch
lr              = 0.0003 # Initial learning rate
momentum        = 0.9 # Momentum
max_norm        = 200 # Norm cutoff to prevent explosion of gradients
l2              = 0 # L2 regularization
batch_size      = 20 # Batch size for training
augment         = True # Use random tempo and gain perturbations
exit_at_acc     = True # Exit at given target accuracy
num_workers     = 4 # Number of workers used in data-loading
cuda            = True # Use cuda to train model
