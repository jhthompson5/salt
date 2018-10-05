import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing
import rle
import random
import pickle
import matplotlib.pyplot as plt
import model

try:
    with open('data/trainSet.dat','rb') as f:
        data = pickle.load(f)
except:
    data = preprocessing.processTrainingData()

zipped = list(zip(data['X1'],data['X2'],data['Y'],data['ids']))
random.shuffle(zipped)
imInputs, depInputs, labels, ids = zip(*zipped)
trainLen = int(0.8*(len(imInputs)))
train = {
    'ids': ids[:trainLen],
    'X1' : np.array(imInputs[:trainLen]),
    'X2' : np.array(depInputs[:trainLen]),
    'Y'  : np.array(labels[:trainLen])
}
test = {
    'ids': ids[trainLen:],
    'X1' : np.array(imInputs[trainLen:]),
    'X2' : np.array(depInputs[trainLen:]),
    'Y'  : np.array(labels[trainLen:])
}

tf.logging.set_verbosity(tf.logging.INFO)

testModel = tf.estimator.Estimator(model_fn=model.conv_nn_model, model_dir='modelSave')


train_fn = tf.estimator.inputs.numpy_input_fn(
    x = train['X1'],
    y = train['Y'],
    batch_size=10,
    num_epochs=None,
    shuffle=False
)  
testModel.train(input_fn=train_fn,steps=3200)

eval_fn = tf.estimator.inputs.numpy_input_fn(
    x = test['X1'],
    y = test['Y'],
    num_epochs=1,
    shuffle=False
)

eval_results = testModel.evaluate(input_fn=eval_fn)
print(eval_results)
