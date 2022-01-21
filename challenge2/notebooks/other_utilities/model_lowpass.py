import os
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from sklearn.metrics import mean_squared_error

window = 600
telescope = 108

order = 6
fs = 1  

# we hand pick the frequency on which to cut for each data
freq = {}
# 'Sponginess'
freq[0] = 0.1
# 'Wonder level'
freq[1] = 0.13
#  'Crunchiness'
freq[2] = 0.13
# 'Loudness on impact'
freq[3] = 0.1
# meme creativity is definitly too smooth to get anythnig useful: let's get just the mean
freq[4] = 0.005
# 'Soap slipperiness'
freq[5] = 0.08
# 'Hype root'
freq[6] = 0.1

folder_model = "inno_training_lowpass_1"

def compute_rmse(x, y):
  rmse = mean_squared_error(x, y, squared=False)
  print("rmse:", rmse)
  return rmse


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def compute_filtered_sig(x, cutoff, fs, order=5, draw=False, title=''):
  b, a = butter_lowpass(cutoff, fs, order)
  w, h = freqz(b, a, worN=8000)
  if draw:
    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title(title + " - Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()

  data = x
  t = np.arange(len(data))
  y = butter_lowpass_filter(data, cutoff, fs, order)
  compute_rmse(x, y)
  if draw:
    plt.subplot(2, 1, 2)
    plt.plot(t, data, 'b-', label='data')
    plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.subplots_adjust(hspace=0.35)
    plt.show()

  return y

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, folder_model))

    def predict(self, X):
        # Insert your preprocessing here
        X = np.array(X)
        X = pd.DataFrame(X)

        dataset = X
        new = {}
        for i in dataset:
          x = dataset[i]
          new[i] = compute_filtered_sig(x, freq[i], fs, order, False, i)
        X = pd.DataFrame.from_dict(new, dtype=np.float64)

        X_min = X.min()
        X_max = X.max()
        print(X_max)
        print(X_min)
        X = (X - X_min)/(X_max - X_min)

        reg_predictions = np.array([])
        d = []
        d.append(X[-window:])
        print(np.array(d).shape)
        X_temp = np.array(d)


        for reg in range(0, 864, telescope):
            pred_temp = self.model.predict(X_temp)
            
            if(len(reg_predictions)==0):
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
            X_temp = np.concatenate((X_temp[:,(telescope):,:],pred_temp), axis=1)

        # Insert your postprocessing here

        t = []
        for j in reg_predictions[0]:
            t.append(j * (X_max - X_min) + X_min)
        reg_predictions = [t]
        
        return tf.constant(reg_predictions[0], dtype='float32')