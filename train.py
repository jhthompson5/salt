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
#zipped = zipped[:int(len(zipped)/2)]
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
'''
train_fn = tf.estimator.inputs.numpy_input_fn(
    x = train['X1'],
    y = train['Y'],
    batch_size=10,
    num_epochs=5,
    shuffle=False
)  
testModel.train(input_fn=train_fn)
'''
eval_fn = tf.estimator.inputs.numpy_input_fn(
    x = test['X1'],
    y = test['Y'],
    num_epochs=1,
    shuffle=False
)

eval_results = testModel.evaluate(input_fn=eval_fn)
print(eval_results)



try:
    with open('data/testSet.dat','rb') as f:
        data = pickle.load(f)
except:
    data = preprocessing.processTestingSet()


data = {
    'X1':np.array(train['X1'][1:3])
}

pred_fn = tf.estimator.inputs.numpy_input_fn(
    x = data['X1'],
    y = np.array(train['Y'][1:3]),
    num_epochs=1,
    shuffle=False
)
print("--------Making Predictions-------\n")
predictions = testModel.predict(input_fn=pred_fn)


p = list(predictions)
for x in p[0:3]:
    print(np.max(x,0))
p_new = [abs(x)/np.max(abs(x),axis=0) for x in p]
p_new = np.reshape(p_new,(2,101,101),'F')
p_new = np.round(p_new)
p_rle = [] 

testP = p_new[:20]
for i in range(len(testP)):
    implt=plt.imshow(testP[i],cmap='gray')
    plt.show()
    print(data['ids'][i])

for i in range(len(p_new)):
    p_rle.append(rle.rle_encode(p_new[i]))
    if i%1000 == 0:
        print("Converting RLEs:",i,"of 18000\n")

preds = {
    'id': data['ids'],
    'rle_mask'  : p_rle
}

df = pd.DataFrame.from_dict(preds)
print(df.head())

#preds = [x for x in predictions]
#print(preds[:10])
df.to_csv("predictions.csv")
