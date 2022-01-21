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
avgwin = 7  # only use odd values
pred_length = 864
telescope = 108


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
        self.model_e1d1 = tf.keras.models.load_model(os.path.join(path, 'model_e1d1'))
        self.model_e2d2 = tf.keras.models.load_model(os.path.join(path, 'model_e2d2'))

    def predict_e1d1(self, X):
        # Insert your preprocessing here
        X = np.array(X)
        X = pd.DataFrame(X)

        new = {}
        for i in X:
            new[i] = getRegression(X[i], size, stride, degree)

        X = pd.DataFrame.from_dict(new)

        X_min = X.min()
        X_max = X.max()
        X = (X - X_min)/(X_max - X_min)

        reg_predictions = np.array([])
        d = []
        d.append(X[-window:])
        X_temp = np.array(d)


        for reg in range(8):
            pred_temp = self.model_e1d1.predict(X_temp)
            
            if(len(reg_predictions)==0):
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
            X_temp = np.concatenate((X_temp[:,(108):,:],pred_temp), axis=1)

        # Insert your postprocessing here

        t = []
        for j in reg_predictions[0]:
            t.append(j * (X_max - X_min) + X_min)
        reg_predictions = np.array([t])
        
        return reg_predictions[0]

        
    def predict_e2d2(self, X):
        # Insert your preprocessing here
        X = np.array(X)
        X = pd.DataFrame(X)
        X_min = X.min()
        X_max = X.max()
        X = (X - X_min) / (X_max - X_min)

        reg_predictions = np.array([])
        reg_predictions_averaged = np.array([])
        d = []
        d.append(X[-window:])
        X_temp = np.array(d)

        for reg in range(int(pred_length / telescope)):
            pred_temp = self.model_e2d2.predict(X_temp)
            if len(reg_predictions) == 0:
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions, pred_temp), axis=1)
            X_temp = np.concatenate((X_temp[:, (telescope):, :], pred_temp), axis=1)

        # Insert your postprocessing here

        # sliding window average
        X2 = X.values
        a = int(np.floor(avgwin / 2))
        reg_predictions_averaged = np.zeros(shape=(reg_predictions.shape))
        for j, x2 in enumerate(reg_predictions[0].T):
            for k, x in enumerate(x2):
                if k < a:
                    reg_predictions_averaged[0, k, j] = sum(
                        np.concatenate(
                            (X2[-(a - k) :, j], reg_predictions[0, : (a + k + 1), j])
                        )
                    ) / (avgwin)
                elif x2.size - k <= a:
                    reg_predictions_averaged[0, k, j] = sum(
                        reg_predictions[0, -(x2.size - k + a) :, j]
                    ) / (reg_predictions[0, -(x2.size - k + a) :, j].size)
                else:
                    reg_predictions_averaged[0, k, j] = sum(
                        reg_predictions[0, k - a : k + a + 1, j]
                    ) / (avgwin)

        t = []
        for j in reg_predictions_averaged[0]:
            t.append(j * (X_max - X_min) + X_min)
        reg_predictions_averaged = np.array([t])
        return reg_predictions_averaged[0]

    def predict(self, X):
      pred_e2d2 = self.predict_e2d2(X)
      pred_e1d1 = self.predict_e1d1(X)

      pred = np.zeros(pred_e2d2.shape)

      pred[:, 0] = pred_e2d2[:, 0]
      pred[:, 1] = pred_e2d2[:, 1]
      pred[:, 2] = pred_e1d1[:, 2]
      pred[:, 3] = pred_e2d2[:, 3]
      pred[:, 4] = pred_e2d2[:, 4]
      pred[:, 5] = pred_e1d1[:, 5]
      pred[:, 6] = pred_e1d1[:, 6]
      
      pred = np.expand_dims(pred, axis=0)

      return tf.constant(pred[0], dtype="float32")

