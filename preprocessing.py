import rle
import numpy as np
import pandas as pd
from matplotlib import pyplot
import pickle
import matplotlib.pyplot as plt


def getTrainSet():
    image_base_filepath = 'data/train/images/'
    train_filepath = 'data/train.csv'
    df = pd.read_csv(train_filepath)
    imgIds = [x for x in df["id"]]
    rle_mask = [x for x in df["rle_mask"]] 
    images = []
    masks = []
    del df
    for i in imgIds:
        file = image_base_filepath+i+'.png'
        img = pyplot.imread(file)
        img = np.dot(img[...,:3],[0.299, 0.587, 0.114])
        img = np.pad(img,((13,14),(13,14)),'constant')
        images.append(img)
    for i in rle_mask:
        if pd.isna(i):
            i=''
        mask = rle.rle_decode(i,(101,101))
        #mask = np.pad(mask,((13,14),(13,14)),'constant')
        masks.append(mask)

    imClean = {
        'id':imgIds,
        'im':images,
        'mask':masks
    }
    return imClean

def processDepths(images):
    ids = list(images['id'])
    file = 'data/depths.csv'
    df = pd.read_csv(file)
    dep = list(df['z'])
    dids = list(df['id'])
    fdep = {
        'id':[],
        'depth':[]
    }
    for i in range(len(dep)):
        if dids[i] in ids:
            fdep['depth'].append(dep[i])
            fdep['id'].append(dids[i])
    del df
    return fdep

def processTrainingData():
    ims = getTrainSet()
    dep = processDepths(ims)
    sortedDepths = []
    iid = ims['id']
    did = dep['id']
    for i in range(len(iid)):
        index = did.index(iid[i])       
        sortedDepths.append(dep['depth'][index])

    dat = {
        'ids' : np.array(iid),
        'X1' : np.array(ims['im']),
        'X2' : np.array(sortedDepths),
        'Y' : np.array(ims['mask'])
        
    }
    print(len(dat['Y'][0]),len(dat['Y'][0][0]))

    pickle.dump(dat,open('data/trainSet.dat','wb'),-1)
    return dat

def processTestingSet():
    import os
    root_path = 'data/test/images/'
    dirList = os.listdir(root_path)
    images = []
    ids = []
    for i in dirList:
        path = root_path+i
        img = pyplot.imread(path)
        img = np.dot(img[...,:3],[0.299, 0.587, 0.114])
        img = np.pad(img,((13,14),(13,14)),'constant')
        images.append(img)
        ids.append(i.replace('.png',''))
    testSet = {
        'id':ids,
        'X':images
    }
    depths = processDepths(testSet)

    sortedDepths = []
    iid = testSet['id']
    did = depths['id']
    for i in range(len(iid)):
        index = did.index(iid[i])       
        sortedDepths.append(depths['depth'][index])

    dat = {
        'ids' : np.array(iid),
        'X1' : np.array(testSet['X']),
        'X2' : np.array(sortedDepths),        
    }
    pickle.dump(dat,open('data/testSet.dat','wb'),-1)
    return dat

processTrainingData()