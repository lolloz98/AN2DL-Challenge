import os
import tensorflow as tf
import numpy as np

window = 300

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'baseline'))

    def predict(self, X):
        
        # Insert your preprocessing here
        reg_predictions = np.array([])
        d = []
        d.append(X[-window:])
        print(np.array(d).shape)
        X_temp = np.array(d)


        for reg in range(16):
            pred_temp = self.model.predict(X_temp)
            if(len(reg_predictions)==0):
                reg_predictions = pred_temp
            else:
                reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
            X_temp = np.concatenate((X_temp[:,(54):,:],pred_temp), axis=1)

        # Insert your postprocessing here

        return tf.constant(reg_predictions[0], dtype='float32')