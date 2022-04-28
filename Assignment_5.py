#Importing Packages
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold 
from sklearn.model_selection import train_test_split, cross_val_score 

#Exercise 1 - See Report for Answers

#Exercise 2
#A
#Loading the Data and Dropping the last Column
wine_data_train = np.loadtxt('redwine_training.txt')
wine_data_test = np.loadtxt('redwine_testing.txt')

print("Before Last Column Drop Train", wine_data_train.shape)
print("Before Last Column Drop Test", wine_data_test.shape)

wineTrain = wine_data_train[:,:-1]
wineTest = wine_data_test[:,:-1]

print("After Last Column Drop Train", wineTrain.shape)
print("After Last Column Drop Test", wineTest.shape)

#Implementing Linear Regression
def multivarlinreg(x,y):
    
    vector = np.ones((len(y),1))
    x = np.concatenate((vector, x), axis = 1)
    w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x), x)), np.transpose(x)), y)
    
    return w 


#B
acidity_train = wineTrain[:,[0]] #Acidity 
wineTrain # All features except quality 
wineQuality = wine_data_train[:,-1] #Just quality

#Running Regression on 1st Feature
w1 = multivarlinreg(acidity_train, wineQuality)
print("Wine Acidity vs. All Features:" , w1)


#C
#Running Regression on all Features 
w2 = multivarlinreg(wineTrain, wineQuality)
print("All Features vs. Wine Quality:", w2 )




#Exercise 3
#A
#Implement Root Means Square Error 
def rmse(f, t):
    rmse = np.sqrt((1/len(f))*sum((f-t)**2)) 
    return rmse

#B
#Build Regression Model with First Feature Train
acidity_test = wineTest[:,[0]]
wineQualityTest = wine_data_test[:,-1] #Just quality

ones = np.ones((len(acidity_test), 1))
x_acidity = np.concatenate((ones, acidity_test), axis = 1)
w_acidity = multivarlinreg(acidity_train, wineQuality) 
f_acidity = np.dot(x_acidity, w1)

print(rmse(f_acidity, wineQualityTest)) 

#C
#Build Regression Model with All Features Train 
x_all_features = np.concatenate((ones, wineTest), axis = 1)
w_all_features = multivarlinreg(wineTrain, wineQuality)
f_all_features = np.dot(x_all_features, w_all_features)

print(rmse(f_all_features, wineQualityTest))


#Exercise 4
#See Report for Answers 



#Exercise 5
pesticide_data_Train = np.loadtxt('IDSWeedCropTrain.csv', delimiter =',')
pesticide_data_Test = np.loadtxt('IDSWeedCropTest.csv', delimiter =',')

xTrain = pesticide_data_Train[:, :-1]
yTrain = pesticide_data_Train[:, -1]
xTest = pesticide_data_Test[:, :-1]
yTest = pesticide_data_Test[:, -1]

#Training the Random Forest Classifier with 50 Trees 

rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(xTrain, yTrain)

#Determining Accuracy with Test Set
accuracy = accuracy_score(yTest, rfc.predict(xTest))
print("Prediction Accuracy of the Random Forest", accuracy) 





#Exercise 6
def f(x):
    return math.e**(-x/2) + 10*x**2

def derivitive_f(x):
    return 20*x - 0.5*math.e**(-x/2)

def plot_tangents(a, x):
  y = derivitive_f(a)*(x - a) + f(a)
  plt.plot(x, y, '--k', linewidth=1.5, zorder=3, alpha=0.8)

def plot_steps(a, x, ax):
  y = derivitive_f(a)*(x - a) + f(a)
  ax.scatter(a, x, zorder=2)
 
y = np.linspace((-2), (+2), 1000)
def gradient_descent_tangent(y,learning_rates, maximum_interation, tolerance):
        initial_point = 1 #initialized starting point
        iterations = 0 #counting the number of iterations
        
        fig, ax = plt.subplots()
        ax.plot(y, f(y), zorder=1, linewidth=2.5,)

        while initial_point > tolerance and iterations < maximum_interation:
            plot_steps(initial_point, f(initial_point), ax)
            plot_tangents(initial_point, y) 
            ax.plot(initial_point, f(initial_point), 'o', markersize=7)
            plt.axhline(color='black')
            plt.axvline(color='black')
            
            previous_point = initial_point
            initial_point = previous_point - learning_rates * derivitive_f(previous_point)  
          
            if iterations < maximum_interation:

                iterations += 1 
        final_minimum = initial_point
        plt.show()
        return(final_minimum)

y = np.linspace((-2), (+2), 1000)
def gradient_descent(y,learning_rates, maximum_interation, tolerance):
        initial_point = 1 #initialized starting point
        iterations = 0 #counting the number of iterations
        
        fig, ax = plt.subplots()
        ax.plot(y, f(y), zorder=1, linewidth=2.5,)

        while initial_point > tolerance and iterations < maximum_interation:
            plot_steps(initial_point, f(initial_point), ax)
            ax.plot(initial_point, f(initial_point), 'o', markersize=7)
            plt.axhline(color='black')
            plt.axvline(color='black')
            
            previous_point = initial_point
            initial_point = previous_point - learning_rates * derivitive_f(previous_point)
          
            if iterations < maximum_interation:

                iterations += 1 
        final_minimum = initial_point
        plt.show()
        return(final_minimum)

y = np.linspace((-2), (+2), 1000)
def gradient_descent_convergence(y,learning_rates, maximum_interation, tolerance):
        initial_point = 1 #initialized starting point
        iterations = 0 #counting the number of iterations

        while initial_point > tolerance and iterations < maximum_interation:
            previous_point = initial_point
            initial_point = previous_point - learning_rates * derivitive_f(previous_point)  
          
            if abs(initial_point - previous_point) < tolerance:
                print("Gradient descent has converged after", {iterations} , "iterations.")

                break

            if iterations < maximum_interation:
                iterations += 1 

        final_minimum = initial_point
        plt.show()
        return(final_minimum)
          
gradient_descent_tangent(y, learning_rates=0.1, maximum_interation=3, tolerance=10e-10)
gradient_descent_tangent(y, learning_rates=0.01, maximum_interation=3, tolerance=10e-10)
gradient_descent_tangent(y, learning_rates=0.001, maximum_interation=3, tolerance=10e-10)
gradient_descent_tangent(y, learning_rates=0.0001, maximum_interation=3, tolerance=10e-10)

gradient_descent(y, learning_rates=0.1, maximum_interation=10, tolerance=10e-10)
gradient_descent(y, learning_rates=0.01, maximum_interation=10, tolerance=10e-10)
gradient_descent(y, learning_rates=0.001, maximum_interation=10, tolerance=10e-10)
gradient_descent(y, learning_rates=0.0001, maximum_interation=10, tolerance=10e-10)

gradient_descent_convergence(y,learning_rates=0.1, maximum_interation=10000, tolerance=10e-10) 
gradient_descent_convergence(y,learning_rates=0.01, maximum_interation=10000, tolerance=10e-10)
gradient_descent_convergence(y,learning_rates=0.001, maximum_interation=10000, tolerance=10e-10)
gradient_descent_convergence(y,learning_rates=0.0001, maximum_interation=10000, tolerance=10e-10)
#Because the steps being taken are so large, the algorithm crashes each time I try to run it. This is simply the convergence without plots. 




#Exercise 7
#Loading Data
iris_train1 = np.loadtxt('Iris2D1_train.txt')
iris1_test1 = np.loadtxt('Iris2D1_test.txt')
iris_train2 = np.loadtxt('Iris2D2_train.txt')
iris_test2 = np.loadtxt('Iris2D2_test.txt')

iris_train1_data = iris_train1[:,[0,1]]
iris_train1_labels = iris_train1[:,2]
iris_train2_data = iris_train2[:,[0,1]]
iris_train2_labels = iris_train2[:,2]

iris_test1_data = iris_test2[:,[0,1]]
iris_test1_labels = iris_test2[:,2]
iris_test2_data = iris_test2[:,[0,1]]
iris_test2_labels = iris_test2[:,2]

#Scatterplot of each  train datasets
cluster0_iris1 = np.where(iris_train1_labels==0)
cluster1_iris1 = np.where(iris_train1_labels==1)
data_matrix1_iris1 = iris_train1_data[cluster0_iris1]
data_matrix2_iris1 = iris_train1_data[cluster1_iris1]

plt.scatter(data_matrix1_iris1[:,0], data_matrix1_iris1[:,1], c='r')
plt.scatter(data_matrix2_iris1[:,0], data_matrix2_iris1[:,1], c='b')
plt.title('Iris 1 Train Dataset')
plt.show()

cluster0_iris2 = np.where(iris_train2_labels==0)
cluster1_iris2 = np.where(iris_train2_labels==1)
data_matrix0_iris2 = iris_train2_data[cluster0_iris2]
data_matrix1_iris2 = iris_train2_data[cluster1_iris2]

plt.scatter(data_matrix0_iris2[:,0], data_matrix0_iris2[:,1], c='r')
plt.scatter(data_matrix1_iris2[:,0], data_matrix1_iris2[:,1], c='b')
plt.title('Iris 2 Train Dataset')
plt.show()


#Training and Test Error
def logistic_function(input):
    result = 1 / (1 + np.exp(-input)) 

    return result

def logistic_LLH(X, y, w):
    N = X.shape[0]  

    return np.log(1+np.exp(-y * (X @ w))).sum()/N

def logistic_gradient(X, y, w):
   
    N = X.shape[0]

    grad = 0 

    for n in range(N):
        grad += ((-1/N) * y[n] * X[n,:]) * logistic_function(-y[n] * np.dot(w, X[n,:]))

    return grad


def logistic_regression(X, y, max_iter, tolerance=1e-5):

    N, d = X.shape
    ones = np.ones((N, 1))
    X = np.concatenate((ones, X), axis=1)

    y = np.array((y - 0.5)*2)

    learning_rate = 0.01

    w = 0.1 * np.random.randn(d + 1)

    value = logistic_LLH(X, y, w)

    iterations = 0
    convergence = False

    E = []

    for iterations in range (max_iter): 
        iterations = iterations + 1

        grad = logistic_gradient(X, y, w)

        v = -grad

        w_new = w + learning_rate * v

        cur_value = logistic_LLH(X, y, w_new)

        if cur_value < value:
            w = w_new
            value = cur_value
            E.append(value)
            learning_rate *= 1.1
        else:
            learning_rate *= 0.9

        g_norm = np.linalg.norm(grad) 

        if g_norm < tolerance:
            convergence = True
            break
        
    if not convergence: 
        print("The descent procedure did not converge after", {max_iter}, "iterations, the last gradient norm was", {g_norm})
    else:
        print("The descent converged after", {iterations}, "iterations with a gradient magnitude of", {g_norm})
    return w


w_iris_train1 = logistic_regression(iris_train1_data, iris_train1_labels, 1000) 
w_iris_test1 = logistic_regression(iris_test1_data, iris_test1_labels, 1000)
w_iris_train2 = logistic_regression(iris_train2_data, iris_train2_labels, 1000)
w_iris_test2 = logistic_regression(iris_test2_data, iris_test2_labels, 1000) 

print ("The three parameters of the affine linear model applied to the Iris 1 Train dataset: %s" % w_iris_train1)
print ("The three parameters of the affine linear model applied to the Iris 1 Test dataset: %s" % w_iris_test1)
print ("The three parameters of the affine linear model applied to the Iris 2 Train dataset: %s" % w_iris_train2)
print ("The three parameters of the affine linear model applied to the Iris 2 Test dataset: %s" % w_iris_test2) 

N_train1 = iris_train1_data.shape[0]
N_test1 = iris_test1_data.shape[0]
N_train2 = iris_train2_data.shape[0]
N_test2 = iris_test2_data.shape[0] 


def logistic_prediction(X, w):

    N = X.shape[0]
    ones = np.ones((N, 1))
    X = np.concatenate((ones, X), axis=1)

    N = X.shape[0]
    P = np.zeros(N)

    for n in range(N):
        argument = np.exp(np.dot(w, X[n, :]))
        prob_i = argument / (1 + argument)
        P[n] = prob_i

    y = np.round(P)

    y = (y - 0.5) * 2

    return P, y

P_iris1_train, y_iris1_train = logistic_prediction(iris_train1_data, w_iris_train1)
P_iris1_test, y_iris1_test = logistic_prediction(iris_test1_data, w_iris_test1)
P_iris2_train, y_iris2_train = logistic_prediction(iris_train2_data, w_iris_train2)
P_iris2_test, y_iris2_test = logistic_prediction(iris_test2_data, w_iris_test2) 

errors1_train = np.sum(np.abs(y_iris1_train - iris_train1_labels)/2)
errors1_test = np.sum(np.abs(y_iris1_test - iris_test1_labels)/2)
errors2_train = np.sum(np.abs(y_iris2_train - iris_train2_labels)/2)
errors2_test = np.sum(np.abs(y_iris2_test - iris_test2_labels)/2)

error_rate1_train = errors1_train/N_train1
error_rate1_test = errors1_test/N_test1
error_rate2_train = errors2_train/N_train2
error_rate2_test = errors2_test/N_train2

print ("Error rate in Iris 1 Train dataset: %s" % round(error_rate1_train, 2)) # 0.33
print ("Error rate in Iris 1 Test dataset: %s" % round(error_rate1_test, 2))   # 0.17
print ("Error rate in Iris 2 Train dataset: %s" % round(error_rate2_train, 2)) # 0.29
print ("Error rate in Iris 2 Test dataset: %s" % round(error_rate2_test, 2))   # 0.07

w_IRIS1 = w_iris_train1

x_IRIS1 = iris_train1_data 
y_IRIS1 = -x_IRIS1 * (w_IRIS1[1]/w_IRIS1[2]) -w_IRIS1[0]/w_IRIS1[2]

x1_IRIS1 = 4
x2_IRIS1 = 9




#Exercise 8 - See Assignment Report for Answers


#Exercise 9
#A
def kmeans_algorithm():
  
  #Loading the data
  minst_digits = np.loadtxt('MNIST_179_digits.txt')
  minst_labels = np.loadtxt('MNIST_179_labels.txt')

  #KMeans with 3 clusters 
  kmeans = KMeans(n_clusters=3).fit(minst_digits)
  labels = kmeans.labels_

  #Plotting the clusters 
  for cluster_center in kmeans.cluster_centers_:
    plot_clusters(cluster_center) 
  
  plt.show() 

  #Looking for the 1s, 7s and 9s, in each cluster
  for l in range(3):
    cluster = minst_labels[np.where(labels == l)[0]]

    cluster1 = np.where(cluster == 1)[0]
    cluster7 = np.where(cluster == 7)[0]
    cluster9 = np.where(cluster == 9)[0]
    print('There are ', round(len(cluster1) / len(cluster) * 100, 3), '% 1s in cluster', l)
    print('There are ', round(len(cluster7) / len(cluster) * 100, 3), '% 7s in cluster', l)
    print('There are ', round(len(cluster9) / len(cluster) * 100, 3), '% 9s in cluster', l) 

#Defining a function for the image plots
def plot_clusters(digit, labeled=True, title="Cluster Title"):
  plt.show()
  fig = plt.imshow(digit.reshape(28, 28)) 
  fig.set_cmap('Purples')
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False) 

kmeans_algorithm()


#B
minst_digits = np.loadtxt('MNIST_179_digits.txt')
minst_labels = np.loadtxt('MNIST_179_labels.txt') 
# Create indices for cross validation
cv = KFold(n_splits =5)
# loop over cross validation folds and determine k best
k = np.array(range(1,50,2))
kmax=[]
for i in k:
    accuracy=[]
    for train, test in cv.split(minst_digits):
        xTrainCV, xTestCV, yTrainCV, yTestCV = minst_digits[train], minst_digits[test], minst_labels[train], minst_labels[test] 
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrainCV, yTrainCV)
        predictCV = knn.predict(xTestCV)
        accuracy_testCV = accuracy_score(yTestCV, predictCV)
        accuracy.append(accuracy_testCV)
    kmax.append([np.average(accuracy)])
print('Kmax', kmax) 
kbest=k[np.argmax(kmax)]
print("K best Value =", kbest) 

#TestAccuracy
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(xTrainCV, yTrainCV)
prediction3 = knn3.predict(xTestCV)

accuracy_test3 = accuracy_score(yTestCV, prediction3)
print("Testing Accuracy:", accuracy_test3)




#Exercise 10

#A
def pca(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean 
    covariance_matrix = np.cov(centered_data.T)
    EigenValues, EigenVectors = np.linalg.eigh(covariance_matrix)
    
    EigenValues = EigenValues[::-1]
    EigenVectors = EigenVectors[:,::-1]
    
    return EigenValues, EigenVectors

EigenValues, EigenVectors = pca(minst_digits)

cumulative_variables = np.cumsum(EigenValues/np.sum(EigenValues))
plt.plot(cumulative_variables)
plt.grid('on')
plt.title('Cumulative Percentage of Variance')
plt.xlabel('PC Index')
plt.ylabel('Variance')
plt.show()
print(cumulative_variables[0:10])


#B
def mds(data, d): 
  EigenValues, EigenVectors = pca(data)
  datamatrix = np.dot(np.array(EigenVectors).T, data.T)

  return datamatrix[:d]

datamatrix_20 = mds(minst_digits.T, 20)

kmeans = KMeans(n_clusters=3).fit(datamatrix_20)
centers20 = kmeans.cluster_centers_
labels20 = kmeans.labels_

for l in range(3):
    cluster = minst_labels[np.where(labels20 == l)[0]]

    cluster1 = np.where(cluster == 1)[0]
    cluster7 = np.where(cluster == 7)[0]
    cluster9 = np.where(cluster == 9)[0]
    print('There are ', round(len(cluster1) / len(cluster) * 100, 3), '% 1s in cluster', l)
    print('There are ', round(len(cluster7) / len(cluster) * 100, 3), '% 7s in cluster', l)
    print('There are ', round(len(cluster9) / len(cluster) * 100, 3), '% 9s in cluster', l) 

for cluster_center in kmeans.cluster_centers_:
    plot_clusters(cluster_center) 

datamatrix_200 = mds(minst_digits.T, 200)

kmeans = KMeans(n_clusters=3).fit(datamatrix_200)
centers200 = kmeans.cluster_centers_
labels200 = kmeans.labels_

for l in range(3):
    cluster = minst_labels[np.where(labels200 == l)[0]]

    cluster1 = np.where(cluster == 1)[0]
    cluster7 = np.where(cluster == 7)[0]
    cluster9 = np.where(cluster == 9)[0]
    print('There are ', round(len(cluster1) / len(cluster) * 100, 3), '% 1s in cluster', l)
    print('There are ', round(len(cluster7) / len(cluster) * 100, 3), '% 7s in cluster', l)
    print('There are ', round(len(cluster9) / len(cluster) * 100, 3), '% 9s in cluster', l) 

for cluster_center in kmeans.cluster_centers_:
    plot_clusters(cluster_center)



#C
# Create indices for cross validation
cv = KFold(n_splits =5)
# loop over cross validation folds and determine k best
k = np.array(range(1,10,2))
kmax=[]
for i in k:
    accuracy=[]
    for train, test in cv.split(datamatrix_20): 
        xTrainCV, xTestCV, yTrainCV, yTestCV = datamatrix_20[train], datamatrix_20[test], minst_labels[train], minst_labels[test] 
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrainCV, yTrainCV)
        predictCV = knn.predict(xTestCV)
        accuracy_testCV = accuracy_score(yTestCV, predictCV)
        accuracy.append(accuracy_testCV)
    kmax.append([np.average(accuracy)])
print('Kmax', kmax) 
kbest=k[np.argmax(kmax)]
print("K best Value =", kbest) 

#TestAccuracy
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(xTrainCV, yTrainCV)
prediction3 = knn3.predict(xTestCV)

accuracy_test3 = accuracy_score(yTestCV, prediction3)
print("Testing Accuracy:", accuracy_test3)


# Create indices for cross validation
cv = KFold(n_splits =5)
# loop over cross validation folds and determine k best
k = np.array(range(1,10,2))
kmax=[]
for i in k:
    accuracy=[]
    for train, test in cv.split(datamatrix_200): 
        xTrainCV, xTestCV, yTrainCV, yTestCV = datamatrix_200[train], datamatrix_200[test], minst_labels[train], minst_labels[test] 
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrainCV, yTrainCV)
        predictCV = knn.predict(xTestCV)
        accuracy_testCV = accuracy_score(yTestCV, predictCV)
        accuracy.append(accuracy_testCV)
    kmax.append([np.average(accuracy)])
print('Kmax', kmax) 
kbest=k[np.argmax(kmax)]
print("K best Value =", kbest) 

#TestAccuracy
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(xTrainCV, yTrainCV)
prediction3 = knn3.predict(xTestCV)

accuracy_test3 = accuracy_score(yTestCV, prediction3)
print("Testing Accuracy:", accuracy_test3)