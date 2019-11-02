import time

import numpy as np

from tool.progressbar import ProgressBar


def loadData(filepath):
    feturelist = [];
    labellist = [];
    fr = open(filepath, 'r')
    for line in fr.readlines():
        curLine = line.strip().split(',')
        # setting label
        if int(curLine[0]) < 5:
            labellist.append(-1)
        else:
            labellist.append(1)

        # getting feture
        feturelist.append([int(num)/255 for num in curLine[1:]])
    return feturelist, labellist

def perceptron(trainData, trainLabel, iter=50):

    print('Start training')
    dataMat = np.mat(trainData)
    lableMat = np.mat(trainLabel).T

    m, n = np.shape(dataMat)
    w = np.zeros((1, np.shape(dataMat)[1]))
    b = 0
    h = 0.0001

    runTimes = iter * m
    pbar = ProgressBar(n_total=runTimes)
    curTime = 0
    for k in range(iter):
        for i in range(m):
            xi = dataMat[i]
            yi = lableMat[i]
            if yi * (w * xi.T + b) <= 0 :
                w = w + h * yi * xi
                b = b + h * yi
            curTime += 1;
            pbar.batch_step(step=curTime, info={}, bar_type='train data')
    return w, b


def test(testData, testLabel, w, b):
    print('Start testing')
    dataMat = np.mat(testData)
    lableMat = np.mat(testLabel).T

    m, n = np.shape(dataMat)

    erroCnt = 0

    for i in range(m):
        xi = dataMat[i]
        yi = lableMat[i]

        result = yi * (w * xi.T + b)
        if result <=0: erroCnt += 1

    accruRate = 1 - (erroCnt / m)
    return accruRate

if __name__ == '__main__':

    start = time.time()

    trainData, trainLabel = loadData('../data/MNIST/mnist_train.csv')
    testData, testLabel = loadData('../data/MNIST/mnist_test.csv')

    w, b = perceptron(trainData, trainLabel, iter=30)

    accruRate = test(testData, testLabel, w, b)

    end = time.time()
    print('Accuracy rate is:', accruRate)
    print('Time span:', end - start)

