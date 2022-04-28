# Importing Packages
import numpy as np
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

# Reading the Data
dataTrain = np.loadtxt("OccupancyTrain.csv" , delimiter=",")
dataTest = np.loadtxt("OccupancyTest.csv" , delimiter=",")

#Splitting Variables and Labels 
xTrain = dataTrain[:,:-1]
yTrain = dataTrain[:,-1]

xTest = dataTest[:,:-1]
yTest = dataTest[:,-1]




#Exercise 1 - Nearest Neighbor Classification 

#Using KNeighbor Classifer and Accuracy Test
#Test
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xTrain, yTrain)
prediction = knn.predict(xTest)
accuracy_test = accuracy_score(yTest, prediction)
print("Test Accuracy:", accuracy_test)

#Train
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xTrain, yTrain)
prediction = knn.predict(xTrain)
accuracy_test = accuracy_score(yTrain, prediction)
print("Train Accuracy:", accuracy_test)




#Exercise 2 - Cross Validation 
# Create indicies for cross validation
cv = KFold(n_splits =5)

# loop over cross validation folds and determine k best
k = np.array([1, 3, 5, 7, 9, 11])
kmax=[]
for i in k:
    accuracy=[]
    for train, test in cv.split(xTrain):
        xTrainCV, xTestCV, yTrainCV, yTestCV = xTrain[train], xTrain[test], yTrain[train], yTrain[test]
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrainCV, yTrainCV)
        predictCV = knn.predict(xTestCV)
        accuracy_testCV = accuracy_score(yTestCV, predictCV)
        accuracy.append(accuracy_testCV)
    kmax.append([np.average(accuracy)])
print(kmax)
kbest=k[np.argmax(kmax)]
print("K best Value =", kbest)





#Exercise 3 - Evaluation of Classification Performance
#Test 
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(xTrain, yTrain)
prediction3 = knn3.predict(xTest)
accuracy_test3 = accuracy_score(yTest, prediction3)
print("Testing Accuracy:", accuracy_test3)

#Train
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(xTrain, yTrain)
prediction3 = knn3.predict(xTrain)
accuracy_test3 = accuracy_score(yTrain, prediction3)
print("Training Accuracy:", accuracy_test3)





#Exercise 4 - Data Normalization
#normalization
scaler = preprocessing.StandardScaler().fit(xTrain)
xTrainN = scaler.transform(xTrain)
xTestN = scaler.transform(xTest)

#determining kbest
cv = KFold(n_splits =5)
k = np.array([1, 3, 5, 7, 9, 11])
kmax=[]
for i in k:
    accuracy=[]
    for train, test in cv.split(xTrainN):
        xTrainCV, xTestCV, yTrainCV, yTestCV = xTrainN[train], xTrainN[test], yTrain[train], yTrain[test]
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrainCV, yTrainCV)
        predictCV = knn.predict(xTestCV)
        accuracy_testCV = accuracy_score(yTestCV, predictCV)
        accuracy.append(accuracy_testCV)
    kmax.append([np.average(accuracy)])
print(kmax)
kbest=k[np.argmax(kmax)]
print("K best Value =", kbest)

#Training and Test Accuracy of Kbest 
#Test 
knnN = KNeighborsClassifier(n_neighbors=11)
knnN.fit(xTrainN, yTrain)
predictionNtest = knnN.predict(xTestN)
accuracy_testNtest = accuracy_score(yTest, predictionNtest)
print("Testing Accuracy:", accuracy_testNtest)

#Train
knnN = KNeighborsClassifier(n_neighbors=11)
knnN.fit(xTrainN, yTrain)
predictionNtrain = knnN.predict(xTrainN)
accuracy_testNtrain = accuracy_score(yTrain, predictionNtrain)
print("Training Accuracy:", accuracy_testNtrain)