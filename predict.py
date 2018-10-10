import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import rle
import preprocessing
import model
import matplotlib.pyplot as plt

try:
    with open('data/testSet.dat','rb') as f:
        data = pickle.load(f)
except:
    data = preprocessing.processTestingSet()

predModel = tf.estimator.Estimator(model_fn=model.conv_nn_model, model_dir='modelSave')

pred_fn = tf.estimator.inputs.numpy_input_fn(
    x = data['X1'],
    num_epochs=1,
    shuffle=False
)
print("--------Making Predictions-------\n")
predictions = predModel.predict(input_fn=pred_fn)
p = list(predictions)
p = np.reshape(p,(18000,101,101),'F')
p = np.round(p)
p_rle = []
"""
testP = p[:20]
for i in range(len(testP)):
    implt=plt.imshow(testP[i],cmap='gray')
    plt.show()
    print(data['ids'][i])
"""

for i in range(len(p)):
    p_rle.append(rle.rle_encode(p[i]))
    if i%1000 == 0:
        print("Converting RLEs:",i,"of 18000\n")

preds = {
    'id': data['ids'],
    'rle_mask'  : p_rle
}

'''
df = pd.DataFrame.from_dict(preds)
print(df.head())
'''

#preds = [x for x in predictions]
#print(preds[:10])
df.to_csv("predictions.csv")