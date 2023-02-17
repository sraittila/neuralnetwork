import random
import numpy as np

def readData(filename):
    data = []
    file = open(filename, "r")
    for row in file:
        dataPoints = row.split(";")
        if dataPoints[0] == '' or dataPoints[0] == 'Date':
            continue
        
        co = float(dataPoints[2].replace(',','.').strip())
        c6h6 = float(dataPoints[5].replace(',','.').strip())
        if co == -200 or c6h6 == -200:
            continue
        
        data.append([np.array([co]),np.array([c6h6])])
    
    return data

def randomTrainingAndTestSet(dataArray):
    random.shuffle(dataArray)
    dataSet = np.array(dataArray)
    xTraining = dataSet[0:5875,0]
    yTraining = dataSet[0:5875,1]
    xTest = dataSet[5875:len(dataSet),0]
    yTest = dataSet[5875:len(dataSet),1]

    return xTraining,yTraining,xTest,yTest
    
def calculateLoss(theta,xtest, ytest):
    
    estimate = theta[0] + theta[1]*xtest
    testloss = np.mean(np.square(ytest - estimate))
    
    return testloss

def gradientDecent(theta, learningRate,epoch,xtraining, ytraining):
    
    for i in range(epoch): 
        estimate = theta[0] + theta[1]*xtraining 
        dThetaZero =  -np.mean(xtraining * (ytraining - estimate))  
        dThetaOne = -np.mean(ytraining - estimate)  
        theta[0] = theta[0] - learningRate*dThetaZero  
        theta[1] = theta[1] - learningRate*dThetaOne
    
    return theta


data = readData("AirQualityUCI.csv")
shuffledData = randomTrainingAndTestSet(data)
theta = np.array([-4,12])
print("Loss: ")
print(calculateLoss(theta,shuffledData[2],shuffledData[3]))
theta = gradientDecent(theta,0.0001, 8000, shuffledData[0], shuffledData[1])
print(theta)
print("Loss: ")
print(calculateLoss(theta,shuffledData[2],shuffledData[3]))






