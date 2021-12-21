import os
import tensorflow as tf
import numpy as np
import pandas as pd

def preproc(data):
  min = data.min()
  max = data.max()
  return (data - min) / (max - min), min, max    

model_path = 'multimodel_2'
data = {}
data['path'] = []
data['window'] = []
data['stride'] = []
data['telescope'] = []
data['data'] = []

# sponginess
data['path'].append(model_path + '/sponginess')
data['window'].append(600)
data['stride'].append(10)
data['telescope'].append(108)

# wonder level + loudness on impact
data['path'].append(model_path + '/wonder_loudness')
data['window'].append(600)
data['stride'].append(10)
data['telescope'].append(108)

# crunchiness + hype root
data['path'].append(model_path + '/crunch_hype')
data['window'].append(600)
data['stride'].append(10)
data['telescope'].append(108)

# meme creativity
data['path'].append(model_path + '/meme_creat')
data['window'].append(600)
data['stride'].append(10)
data['telescope'].append(108)

# soap slipperiness
data['path'].append(model_path + '/soap_slip')
data['window'].append(600)
data['stride'].append(10)
data['telescope'].append(108)

class model:
    def __init__(self, path):
        self.models = []
        for i in data['path']:
            model = tf.keras.models.load_model(os.path.join(path, i))
            self.models.append(model)

    def predict(self, X):
        X = np.array(X)
        X = pd.DataFrame(X)

        inputs = []
        # Insert your preprocessing here
        inputs.append(preproc(X[0]))
        inputs.append(preproc(X[[1, 3]]))
        inputs.append(preproc(X[[2, 6]]))
        inputs.append(preproc(X[4]))
        inputs.append(preproc(X[5]))

        outputs = []
        for i in range(len(self.models)):
            reg_predictions = np.array([])
            d = []
            if i == 1 or i == 2:
                d.append(inputs[i][0][-data['window'][i]:])
            else:
                d.append([[j] for j in inputs[i][0][-data['window'][i]:]])
            print(np.array(d).shape)
            X_temp = np.array(d)

            for reg in range(864 // data['telescope'][i]):
                pred_temp = self.models[i].predict(X_temp)
                
                if(len(reg_predictions)==0):
                    reg_predictions = pred_temp
                else:
                    reg_predictions = np.concatenate((reg_predictions,pred_temp),axis=1)
                X_temp = np.concatenate((X_temp[:,(data['telescope'][i]):,:],pred_temp), axis=1)
            
            print(reg_predictions.shape)

            t = []
            for j in reg_predictions[0]:
                t.append(j * (inputs[i][2] - inputs[i][1]) + inputs[i][1])
            reg_predictions = np.array(t)
            print('shape: ', reg_predictions.shape)
            if reg_predictions.shape[1] == 2:
                outputs_tmp = [[], []]
                for i in reg_predictions:
                    outputs_tmp[0].append([i[0]])
                    outputs_tmp[1].append([i[1]])
                print('shape tmp0: ',np.array(outputs_tmp).shape)
                outputs.append(outputs_tmp[0])
                outputs.append(outputs_tmp[1])
            else:  
                outputs.append(reg_predictions)

        # 0: sponginess ->   0
        # 1: wonder level -> 1
        # 2: loudness ->     3
        # 3: crunch          2
        # 4: hype            6
        # 5: meme            4
        # 6: soap            5
        output = np.array([np.concatenate((outputs[0], outputs[1], outputs[3], outputs[2], outputs[5], outputs[6], outputs[4]), axis=1)])
        print(output.shape)

        return tf.constant(output[0], dtype='float32')