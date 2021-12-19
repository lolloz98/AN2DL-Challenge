import os
import tensorflow as tf
import numpy as np
import pandas as pd

window = 300
avgwin = 11  # only use odd values
pred_length = 864
telescope = 54


class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, "model"))

    def predict(self, X):
        # Insert your preprocessing here

        X = np.array(X)
        X = pd.DataFrame(X)

        # input smoothing

        data = X.values
        data_smooth = np.zeros(shape=data.shape)
        a = int(np.floor(avgwin / 2))
        tmp = np.array(0)
        for j, x in enumerate(data.T):  # x is every one of the 7 timeseries
            print(j)
            for k, s in enumerate(x):  # s is every point of the single time serie
                sum = 0
                count = 0
                if k < a:
                    tmp = x[: (a + k + 1)]
                elif x.size - k <= a:
                    tmp = x[-(x.size - k + a) :]
                else:
                    tmp = x[k - a : k + a + 1]
                if (tmp == np.ones(shape=tmp.shape)).all():
                    sum = 1
                    count = 1
                    # print("error") 
                    # this happens when all points in the window are ones
                    # it would be nice to implement for the moment something like picking a wider window
                    # but for now i'm leaving them as ones
                for x1 in tmp:
                    if x1 != 1:
                        sum = sum + x1
                        count = count + 1
                data_smooth[k, j] = sum / count

        X = pd.DataFrame(data_smooth, columns = X.columns)

        # input smoothing over 

        X_min = X.min()
        X_max = X.max()
        print(X_max)
        print(X_min)
        X = (X - X_min) / (X_max - X_min)

        reg_predictions = np.array([])
        d = []
        d.append(X[-window:])
        print(np.array(d).shape)
        X_temp = np.array(d)

        for reg in range(int(pred_length / telescope)):
            pred_temp = self.model.predict(X_temp)
            if len(reg_predictions) == 0:
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions, pred_temp), axis=1)
            X_temp = np.concatenate((X_temp[:, (telescope):, :], pred_temp), axis=1)

        # Insert your postprocessing here

        t = []
        for j in reg_predictions[0]:
            t.append(j * (X_max - X_min) + X_min)
        reg_predictions = [t]

        return tf.constant(reg_predictions[0], dtype="float32")
