#@title Import libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
plt.rc('font', size=16)
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

tfk = tf.keras
tfkl = tf.keras.layers
# print(tf.__version__)
#@title init seed everywhere
seed =20

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

dataset_path = '../SharingFolder/Training.csv'
dataset  = pd.read_csv(dataset_path)
print(dataset.shape)
npds = np.array(dataset)

# print(dataset)

from model import model
a = model("./")


# print(dataset[-864:-863])

pred = a.predict(npds[:-864])

err = [0 for i in range(7)]
for i, p in enumerate(pred):
    err += (p - np.array(dataset)[-864 + i]) ** 2
err /= 864
err = err ** 0.5
print(err)

def inspect_multivariate_prediction(X, y, pred, columns, telescope, idx=None):
    if(idx==None):
        idx=np.random.randint(0,len(X))

    figs, axs = plt.subplots(len(columns), 1, sharex=True, figsize=(17,17))
    for i, col in enumerate(columns):
        axs[i].plot(np.arange(len(X[0,:,i])), X[idx,:,i])
        axs[i].plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), y[idx,:,i], color='orange')
        axs[i].plot(np.arange(len(X[0,:,i]), len(X[0,:,i])+telescope), pred[idx,:,i], color='green')
        axs[i].set_title(col)
    plt.show()

X = np.array([npds[(-600-864):-864]])
y = np.array([npds[-864:]])
pred = np.array([pred])
columns = dataset.columns
telescope = 864
idx = 0
inspect_multivariate_prediction(X, y, pred, columns, telescope, idx)