#pip3 install tensorflow
#pip3 install --upgrade pip
#pip3 install --upgrade tensorflow

#Python 3.6.9
#TensorFlow 2.6.0

#pip3 install sounddevice
#pip3 install wavio

#sudo apt-get install libportaudio2

#pip3 install tqdm


import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

import time
from tqdm import tqdm

import logging
import threading
import subprocess

import wavio as wv
import sounddevice as sd
from scipy.io.wavfile import read,write

# Creating status bar while uploading a model
def thread_function(name):
    logging.info("Load    : Loading Custom Model...")
    global model 
    model = tf.keras.models.load_model('ResNet152')
    logging.info("Load    : Done.")

# Audio to tensor convertion 
def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)

  '''
  audio = tf.Tensor(
  [[ 0.00057983]
  [ 0.00076294]
  [ 0.00100708]
  ...
  [-0.00085449]
  [-0.00067139]
  [-0.00054932]], shape=(16000, 1), dtype=float32)
  '''
  return tf.squeeze(audio, axis=-1)

  '''
  tf.Tensor(
  [ 0.00057983  0.00076294  0.00100708 ... -0.00085449 -0.00067139
    -0.00054932], shape=(16000,), dtype=float32)

  '''

# Audio Files to tensors mapping function 
def get_waveform_and(file_path):
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform

# Waveform to mfcc mapping function 
def get_mfcc(waveform):
  
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  
  # The STFT (tf.signal.stft) splits the signal into windows of time 
  # and runs a Fourier transform on each window
  frame_length = 255 
  stfts = tf.signal.stft(
      equal_length, frame_length=frame_length, frame_step=128)
  
  ## FFT_SIZE = 255
  ## 16000 / 128 = 125 windows

  # STFT produces an array of complex numbers representing magnitude and phase. 
  # However, we need only  the magnitude 
  
  spectr = (1/frame_length)*tf.pow(tf.abs(stfts),2)
  #spectrogram = tf.abs(stfts) PREVIOUS

  ## Replace zero values with something else:

  spectrograms = tf.where(spectr > 0, spectr, 10**(-2))
  sample_rate = 16000

  # So far we have just compute spectrogram (as get_spectrogram())
  # With the next commands, mfcc's are computed 
  

  # bins = FFT_SIZE/2 + 1 = 255/2 + 1 = 129
  num_spectrogram_bins = stfts.shape[-1]

  # min and max frequency in hertz (will be convertet to mel)
  # num_mel_bins = number of triangle filters 
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80

  #  linear_to_mel_weight_matrix: array M=(FFT_SIZE/2 + 1, #num_mel_bins)

  ## 1. lower_edge_hertz and upper_edge_hertz are converted from Hz to Mel 
  ## 2. The interval between these points is devided into (num_mel_bins+1) parts
  ## 3. Bounds of these parts are converted from Mel to Hz again 
  ## 4. These bounds are rounded to the nearest frequency bin
  ## 5. Triangle filters are constructed so as 
    ##  the left  bound of one filter being the center of previous one
    ##  the right bound of one filter being the center of next one (in Mels)

  ## So each Column is simply one triangle filter

  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
  
  #plt.plot(linear_to_mel_weight_matrix)

  # Remember: spectrogram is array Y = (#FRAMES, FFT_SIZE/2+1)
  mel_spectrograms = tf.tensordot(
    spectrograms, linear_to_mel_weight_matrix, 1)
    
  # So Mel_Spectrogram = Y*M 
  # = (#FRAMES, FFT_SIZE/2+1) * (#FFT_SIZE/2 + 1,#num_mel_bins) 
  # = (#FRAMES, #num_mel_bins)

  # spectrograms.shape[:-1] = #FRAMES
  # linear_to_mel_weight_matrix.shape[-1:] = #num_mel_bins

  #Ensures that shape is compatible 
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
    linear_to_mel_weight_matrix.shape[-1:]))
  
  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  # Mel_Spectrogram = log(Y*M) 
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  # For each frame (line), we compute DCT 
  # After this we keep 13 first columns thata are MFCC's
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms)[..., :13]

  return mfccs

# Insert new axes (expand)
def get_mfcc_and_expand(audio):
  mfcc = get_mfcc(audio)
  mfcc = tf.expand_dims(mfcc, -1)
  return mfcc

# Waveform to delta-mfccs mapping function 
def get_delta_mfccs(mfccs, N=1):  
  #For each frame, calculate delta features 
  #based on preceding and following N frames
  numframes = len(mfccs)
  coefficients = mfccs.shape[1]

  # Shape of mfcc's is (numframes,coefficients,1)
  # so is turned to (numframes,coefficients)
  mfccs = tf.reshape(mfccs,[numframes,coefficients])

  #Padding edge values of mfcc's vertically 
  paddings_1 = tf.constant([[N, 0,], [0, 0]])
  padded_mfccs_1 = tf.pad(mfccs, paddings_1, "SYMMETRIC")
  x1 = -N * padded_mfccs_1[:-N]

  #Padding edge values of mfcc's vertically 
  paddings_2 = tf.constant([[N-1, 0,], [0, 0]])
  padded_mfccs_2 = tf.pad(mfccs, paddings_2, "SYMMETRIC")  
  x2 = -(N-1) * padded_mfccs_2[:-(N-1)]

  #Padding edge values of mfcc's vertically 
  paddings_3 = tf.constant([[0, N-1,], [0, 0]])
  padded_mfccs_3 = tf.pad(mfccs, paddings_3, "SYMMETRIC")  
  x3 = (N-1)* padded_mfccs_3[N-1:]

  #Padding edge values of mfcc's vertically 
  paddings_4 = tf.constant([[0, N,], [0, 0]])
  padded_mfccs_4 = tf.pad(mfccs, paddings_4, "SYMMETRIC")  
  x4 = N * padded_mfccs_4[N:]


  denominator = 2 * sum([i**2 for i in range(1, N+1)])

  if (N == 2):
    delta_features = (x1+x2+x3+x4) / denominator
  if (N == 1):
    delta_features = (x1+x4) / denominator
  return delta_features

# Insert new axes (expand)
def get_delta_mfcc_and_expand(mfcc):
  delta_mfcc = get_delta_mfccs(mfcc)
  delta_mfcc = tf.expand_dims(delta_mfcc, -1)
  return delta_mfcc

# Preprocess dataset (Call the above mapping functions)
def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and)
  mfcc_ds = output_ds.map(get_mfcc_and_expand)
  output_ds = mfcc_ds.map(get_delta_mfcc_and_expand)
  return output_ds


# Step 1: Loading Custom Model
logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO,
                    datefmt="%H:%M:%S")

x = threading.Thread(target=thread_function, args=(1,))
x.start()
for i in tqdm(range(65)):
    if not x.is_alive():
        break
    time.sleep(1)
x.join()
logging.info("Main    : end.")
print()
print()



# Step 2: Recording voice
# Sampling frequency
freq = 16000

# Recording duration
duration = 1

while(True):
    # Start recorder with the given values of 
    # duration and sample frequency
    print("Recording    : Starting Recording...")
    print("Recording    : Fs = 16000 Hz")
    print("Recording    : Duration = 1sec")

    key_pad = input("Recording    : Press ENTER key and start recording ...")
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1, dtype = 'int16')
    sd.wait()
    print("Recording    : Done")
    print("Recording    : my_own.wav was saved")
    # Record audio for the given number of seconds
    

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("my_own.wav", freq, recording)


    # Save your voice
    sample_file = pathlib.Path("my_own.wav")
    print()

    print("Feature Extraction: Starting ...")
    sample_ds = preprocess_dataset([str(sample_file)])
    print("Feature Extraction: delta mfccs are done.")
    print()

    commands= np.array(['left','right','go','stop','yes','no','up','down','one','two','three','four'])

    for features in sample_ds.batch(1):
        prediction = model(features)
        probs = tf.nn.softmax(prediction[0])
        arg_max_prob = tf.math.argmax(probs).numpy()

        # Uncomment for softmax details 
        
        #fmt = '{:<5} {:<1} {:<2} {:<2}'
        #for i in range(len(commands)):
        #    prob = round(probs.numpy()[i]*100,2)
        #    print(fmt.format(commands[i],':', prob,'%'))
        
        #print(probs.numpy()[arg_max_prob])

        if (probs.numpy()[arg_max_prob] < 0.1):
            print("Output       : Unknown")
        else:
            print("Output       :", commands[arg_max_prob])
        print()


