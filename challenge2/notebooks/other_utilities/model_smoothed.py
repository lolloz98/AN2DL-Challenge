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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

window = 600
size = 80
stride = 50
degree = (8, 10)

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

def getRegForChunk(chunk, degree, splits_for_k_val=1, draw=False):
  """
  chunk: a 1D array of data
  degree: a tuple, (min_degree, max_degree)
  draw: draws the generated regression
  """
  X = np.arange(len(chunk)).reshape(-1, 1)

  # compute best degree
  best_score = 10000000
  best_deg = 5
  for d in range(degree[0], degree[1] + 1):
    poly = PolynomialFeatures(degree=d)
    poly_features = poly.fit_transform(X)
    poly_reg_model = linear_model.LinearRegression()

    if splits_for_k_val > 1:
      score = cross_val_score(poly_reg_model, poly_features, chunk, cv=KFold(n_splits=splits_for_k_val, shuffle=True, random_state=1234))
      if score.mean() < best_score:
        best_score = score.mean()
        best_deg = d
    else:
      poly_reg_model.fit(poly_features, chunk)
      y_predicted = poly_reg_model.predict(poly_features)
      r2 = r2_score(chunk, y_predicted)
      if r2 < best_score:
        best_score = r2
        best_deg = d

  # get final model
  poly = PolynomialFeatures(degree=best_deg)
  poly_features = poly.fit_transform(X)
  poly_reg_model = linear_model.LinearRegression()
  poly_reg_model.fit(poly_features, chunk)
  y_predicted = poly_reg_model.predict(poly_features)
  
  if draw:
    plt.figure(figsize=(10, 6))
    plt.scatter(X, chunk)
    plt.plot(X, y_predicted, c='red')
    plt.show()
  
  return y_predicted

def allChunks(chunks, degree, k_val_splits, draw=False):
  ret = []
  for c in chunks:
    ret.append(getRegForChunk(c, degree, splits_for_k_val=k_val_splits, draw=draw))
  return ret

def getRegression(x, size, stride, degree, k_val_splits=1):
  y = chunk(x, size, stride)
  y = allChunks(y, degree, k_val_splits)
  r = np.array(rebuild(y, size, stride))
  return r


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'inno_training_smooth_4'))

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