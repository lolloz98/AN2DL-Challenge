import os
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

window = 600
size = 60
stride = 45
degree = 15

def chunk(x, size, stride):
  """
  divide the dataset in chunks. It keeps the right-most elements and discard the left ones, if size and stride are not
  compatible with dimensions
  """
  ret = []
  for i in range(len(x) - size, -1, -stride):
    ret.append(x[i: i + size])
  ret.reverse()
  return ret

def rebuild(y, size, stride):
  """
  rebuild the original (if no cut were made) after the chunk function
  """
  s = size - stride
  ret = [0 for i in range(len(y[0]) * len(y) - (s * (len(y) - 1)))]

  for i in range(len(y[0])):
      ret[i] = y[0][i]

  ind = size
  for i in y[1:]:
    l = s
    for j in i:
      if l > 0:
        ret[ind - l] = (ret[ind - l] + j) / 2
      else:
        ret[ind - l] = j
      l -= 1
    ind += size - s
  return ret

def getRegForChunk(chunk, degree, draw=False):
  """
  chunk: a 1D array of data
  degree: a single number, max_degree
  draw: draws the generated regression
  """
  X_ax = np.arange(len(chunk)).reshape(-1, 1)

  poly = PolynomialFeatures(degree=degree)
  poly_features = poly.fit_transform(X_ax)
  poly_reg_model = linear_model.LinearRegression()
  poly_reg_model.fit(poly_features, chunk)
  y_predicted = poly_reg_model.predict(poly_features)
  
  if draw:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_ax, chunk)
    plt.plot(X_ax, y_predicted, c='red')
    plt.show()
  
  return y_predicted


def allChunks(chunks, degree, draw=False):
  ret = []
  for c in chunks:
    ret.append(getRegForChunk(c, degree, draw))
  return ret

def getRegression(x, size, stride, degree):
  y = chunk(x, size, stride)
  y = allChunks(y, degree)
  r = np.array(rebuild(y, size, stride))
  return r


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'inno_training_smooth_2'))

    def predict(self, X):
        # Insert your preprocessing here
        X = np.array(X)
        X = pd.DataFrame(X)

        new = {}
        for i in X:
            new[i] = getRegression(X[i], size, stride, degree)

        X = pd.DataFrame.from_dict(new)

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


        for reg in range(8):
            pred_temp = self.model.predict(X_temp)
            
            if(len(reg_predictions)==0):
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
            X_temp = np.concatenate((X_temp[:,(108):,:],pred_temp), axis=1)

        # Insert your postprocessing here

        t = []
        for j in reg_predictions[0]:
            t.append(j * (X_max - X_min) + X_min)
        reg_predictions = [t]
        
        return tf.constant(reg_predictions[0], dtype='float32')