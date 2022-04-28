#Importing Packages 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import linalg as lg
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

#Exercise 1 - Performing PCA
#A
#Importing Datasets
datam = np.loadtxt('murderdata2d.txt')
def pca(datam):
    mean = np.mean(datam, axis=0)
    centered_data = datam - mean 
    covariance_matrix = np.cov(centered_data.T)
    EigenValues, EigenVectors = np.linalg.eigh(covariance_matrix)
    
    EigenValues = EigenValues[::-1]
    EigenVectors = EigenVectors[:,::-1]
    
    return EigenValues, EigenVectors

EigenValues, EigenVectors = pca(centered_data)

#B

#Plotting Scatter Plot 
plt.scatter(centered_data[:,0], centered_data[:,1])

#Compute mean
mean0 = np.mean(centered_data, axis=0)
mean1 = np.mean(centered_data)
# Compute the corresponding standard deviations
s0 = np.sqrt(EigenValues[0])
s1 = np.sqrt(EigenValues[1])

plt.plot(mean0[0],mean0[1], 'x')
plt.plot([mean1, mean1 + s0*EigenVectors[0,0]], [mean1, mean1 + s0*EigenVectors[1,0]], 'r')
plt.plot([mean1, mean1 + s1*EigenVectors[0,1]], [mean1, mean1 + s1*EigenVectors[1,1]], 'r')
plt.title('Murder Rates Correlated with Unemployment Rates')
plt.xlabel('Unemployment Rate (%)') 
plt.ylabel('Murders per Year per 1.000.000 inhabitants')
plt.show() 

#C
#Reading the Data
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
#Splitting Input Variables and Labels 
XTrain = dataTrain[:,:-1]
YTrain = dataTrain[:,-1]
XTest = dataTest[:,:-1]
YTest = dataTest[:,-1]

datap = np.vstack((XTrain, XTest))
print(datap.shape)

EigenValues, EigenVectors = pca(datap)

plt.plot(range(0,len(EigenValues)),EigenValues)
plt.xlabel('PCs in descending order')
plt.ylabel('Projected variance')
plt.title('PCA on Pesticide Data')
plt.grid('on')
plt.show()

#D
#Loading the data and dropping the column
datao = np.loadtxt('occupancy_data.csv', delimiter=',')
print("Before Last Column Drop", datao.shape)
df = pd.DataFrame(datao)
datao_adjusted = df.drop(df.columns[6], axis=1)
print("After Last Column Drop", datao_adjusted.shape)

EigenValues , EigenVectors = pca(datao_adjusted)

plt.plot(EigenValues)
plt.grid('on')
plt.xlabel('PCs in descending order')
plt.ylabel('Accumulated projected variance')
plt.title('PCA on Occupancy Data - Unnormalized')
plt.show()

#E
scaler = preprocessing.StandardScaler().fit(datao_adjusted)
scaled_occupancy_data = scaler.transform(datao_adjusted)
EigenValues, EigenVectors = pca(scaled_occupancy_data)
plt.plot(EigenValues)
plt.grid('on')
plt.xlabel('PCs in descending order')
plt.ylabel('Accumulated projected variance')
plt.title('PCA on Occupancy Data - Standardized')
plt.show() 

scaler = preprocessing.StandardScaler().fit(datao_adjusted)
norm_data = scaler.transform(datao_adjusted)
EigenValues, EigenVectors = pca(norm_data)
c_var = np.cumsum(EigenValues/np.sum(EigenValues))
plt.plot(c_var)
plt.grid('on')
plt.title('Cumulative Explained Varience')
plt.xlabel('Number of Features')
plt.ylabel('Accumulated projected variance')
plt.show()
print(c_var[0:10])

#Exercise 2 - Visualization in 2D
#A

def mds(data, d): 
  EigenValues, EigenVectors = pca(data)
  datamatrix = np.dot(np.array(EigenVectors).T, data.T)

  return datamatrix[:d]


datamatrix = mds(datap, 2)
x = [datamatrix[0, :]]
y = [datamatrix[1, :]]
plt.scatter(x , y )
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("First 2 PCs of Pesticide Data")
plt.show()

#B

#Unnormalized Data

datamatrix = mds(datao_adjusted, 2) 

plt.scatter(datamatrix[0,:], datamatrix[1,:])
plt.grid('on')
plt.title('Projected Unnormalized Data - First 2 PCs')
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.axis('equal')
plt.show()

#Normalized Data
datamatrix = mds(scaled_occupancy_data, 2)

plt.scatter(datamatrix[0,:], datamatrix[1,:])
plt.grid('on')
plt.title('Projected Normalized Data - First 2 PCs')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#Exercise 3 - Clustering 

startingPoint = np.vstack((XTrain[0,],XTrain[1,]))
kmeans = KMeans(n_clusters=2, algorithm='full', n_init=1, init=startingPoint).fit(XTrain) 
print("Clusters 1 and 2", kmeans.cluster_centers_)

#Exercise 4 - Bayesian Statistics
#Answers in report 