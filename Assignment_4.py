#Importing Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.cluster import KMeans



#Exercise 1 - Plotting Cell Shapes
data_diatoms = np.loadtxt('diatoms.txt')

sequence = np.arange(780*182,dtype='float64').reshape(780,182)

sequence[:, :-2] = data_diatoms

sequence[:, -2] = data_diatoms[:, 0]
sequence[:, -1] = data_diatoms[:, 1]
print(data_diatoms.shape)
#Plotting One Cell *** write function to close the loop, like faces from class ex
x = sequence[:,0::2]
y = sequence[:,1::2]

plt.plot(x[0],y[0])
plt.axis('equal')
plt.title('Plot of One Diatom Cell')
plt.show()

#Plotting All Cells *** write function to close the loop, like faces from class ex
for i in range(data_diatoms.shape[0]):
    plt.plot(x[i,:],y[i,:])
plt.axis('equal')
plt.title('Plot of Many Diatom Cells - Each Plotted On Top of Previous Cell')
plt.show()



#Exercise 2 - Visualizing Variance in Visual Data
def pca(data):
    mean = np.mean(data, axis=0)
    centered_data = data - mean 
    covariance_matrix = np.cov(centered_data.T)
    EigenValues, EigenVectors = np.linalg.eigh(covariance_matrix)
    
    EigenValues = EigenValues[::-1]
    EigenVectors = EigenVectors[:,::-1]
    
    return EigenValues, EigenVectors

EigenValues , EigenVectors = pca(sequence)
mean = np.mean(sequence, axis = 0)


A = [0,1,2]
new =[]
#append loop to 0,1,3 
for i in A: 
        CellOne = mean - (2*(EigenValues[i]**0.5)*EigenVectors[:,i])
        new.append(CellOne)
        CellTwo = mean - ((EigenValues[i]**0.5)*EigenVectors[:,i])
        new.append(CellTwo)
        CellThree = mean
        new.append(CellThree)
        CellFour = mean + ((EigenValues[i]**0.5)*EigenVectors[:,i])
        new.append(CellFour)
        CellFive = mean + (2*(EigenValues[i]**0.5)*EigenVectors[:,i])
        new.append(CellFive)
def plotxy(A): 
    x = A[0::2]
    y = A[1::2]

    return(x,y)

CellOneX  , CellOneY = plotxy(new[0]) 
CellTwoX , CellTwoY = plotxy(new[1])
CellThreeX , CellThreeY = plotxy(new[2])
CellFourX , CellFourY = plotxy(new[3])
CellFiveX , CellFiveY = plotxy(new[4])

plt.plot(CellOneX, CellOneY)
plt.plot(CellTwoX , CellTwoY)
plt.plot(CellThreeX , CellThreeY)
plt.plot(CellFourX , CellFourY)
plt.plot(CellFiveX , CellFiveY)
plt.axis('equal')
plt.title('Variance Over First PC')
plt.show()



CellOneX  , CellOneY = plotxy(new[5]) 
CellTwoX , CellTwoY = plotxy(new[6])
CellThreeX , CellThreeY = plotxy(new[7])
CellFourX , CellFourY = plotxy(new[8])
CellFiveX , CellFiveY = plotxy(new[9])

plt.plot(CellOneX, CellOneY)
plt.plot(CellTwoX , CellTwoY)
plt.plot(CellThreeX , CellThreeY)
plt.plot(CellFourX , CellFourY)
plt.plot(CellFiveX , CellFiveY)
plt.axis('equal')
plt.title('Variance Over Second PC')
plt.show()



CellOneX  , CellOneY = plotxy(new[10]) 
CellTwoX , CellTwoY = plotxy(new[11])
CellThreeX , CellThreeY = plotxy(new[12])
CellFourX , CellFourY = plotxy(new[13])
CellFiveX , CellFiveY = plotxy(new[14])

plt.plot(CellOneX, CellOneY)
plt.plot(CellTwoX , CellTwoY)
plt.plot(CellThreeX , CellThreeY)
plt.plot(CellFourX , CellFourY)
plt.plot(CellFiveX , CellFiveY)
plt.axis('equal')
plt.title('Variance Over Third PC')
plt.show()




#Exercise 3 A -- See Report for Answers



#B

data_toy = np.loadtxt('pca_toydata.txt')
EigenValues , EigenVectors = pca(data_toy)

def mds(data, d): 
  EigenValues, EigenVectors = pca(data)
  principlecomponents = EigenVectors[:,:2]
  datamatrix = np.dot(np.array(EigenVectors).T, data.T)

  return datamatrix[:d]

#Projection onto First 2 PCs

datamatrix = mds(data_toy, 2)
x = [datamatrix[0, :]]
y = [datamatrix[1, :]]
plt.scatter(x , y )
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("First 2 PCs of Toy Data")
plt.show()

#Projection onto First 2 PCs without Last 2 Datapoints
#Dropping last two points
print("Before Row Drop", data_toy.shape)
df = pd.DataFrame(data_toy)
data_toy_adjusted = df.drop(df.index[499:501], axis = 0)
print("After Row Drop", data_toy_adjusted.shape)

datamatrix = mds(data_toy_adjusted, 2)
x = [datamatrix[0, :]]
y = [datamatrix[1, :]]
plt.scatter(x , y )
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("First 2 PCs of Toy Data After 2 Rows Drop")
plt.show()



#Exercise 4 - Clustering II

dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
lables = dataTrain[:,-1]
dataX = dataTrain[:,:-1]

#Initialize and Compute Centroids
startingPoint = np.vstack((dataX[0,],dataX[1,]))
kmeans = KMeans(n_clusters=2, algorithm='full', n_init=1, init=startingPoint).fit(dataX) 
print("Clusters 1 and 2", kmeans.cluster_centers_)

#Splitting Clusters 1 and 2 
centroids = kmeans.cluster_centers_
center1 = centroids[0]
center2 = centroids[1]

#PCA
EigenValues , EigenVectors = pca(dataX)

pc1_data = np.dot(dataX,EigenVectors[:,0])
pc2_data = np.dot(dataX,EigenVectors[:,1])


pc1_center = np.dot(center1,EigenVectors[:,0])
pc2_center = np.dot(center1,EigenVectors[:,1])

pc1_center2 = np.dot(center2,EigenVectors[:,0])
pc2_center2 = np.dot(center2,EigenVectors[:,1])

for i in range (len(lables)):
    if lables[i] == 0:
        dataX[i]
        plt.scatter(pc1_data[i], pc2_data[i], color = 'pink', s=5)
    else: 
        plt.scatter(pc1_data[i], pc2_data[i], color = 'slateblue', s=5)
plt.scatter(pc1_center, pc2_center, marker = 'X', color = 'black')
plt.scatter(pc1_center2, pc2_center2, marker ='X', color = 'black')
plt.title('Projection of Pesticide Data and Cluster Centers')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()



#Exercise 5
dataOccupancy = np.loadtxt('occupancy_data.csv', delimiter=',')
lables = dataOccupancy[:,-1]
dataX = dataOccupancy[:,:-1]

#Initialize and Compute Centroids
startingPoint = np.vstack((dataX[0,],dataX[1,]))
kmeans = KMeans(n_clusters=2, algorithm='full', n_init=1, init=startingPoint).fit(dataX) 
print("Clusters 1 and 2", kmeans.cluster_centers_)

#Splitting Clusters 1 and 2 
centroids = kmeans.cluster_centers_
center1 = centroids[0]
center2 = centroids[1]

#PCA
EigenValues , EigenVectors = pca(dataX)

pc1_data = np.dot(dataX,EigenVectors[:,0])
pc2_data = np.dot(dataX,EigenVectors[:,1])

pc1_center = np.dot(center1,EigenVectors[:,0])
pc2_center = np.dot(center1,EigenVectors[:,1])

pc1_center2 = np.dot(center2,EigenVectors[:,0])
pc2_center2 = np.dot(center2,EigenVectors[:,1])

for i in range (len(lables)):
    if lables[i] == 0:
        dataX[i]
        plt.scatter(pc1_data[i], pc2_data[i], color = 'pink', s=5)
    else: 
        plt.scatter(pc1_data[i], pc2_data[i], color = 'purple', s=5)
plt.scatter(pc1_center, pc2_center, marker = 'X', color = 'black')
plt.scatter(pc1_center2, pc2_center2, marker ='X', color = 'black')
plt.title('Projection of Occupancy Data and Cluster Centers')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show() 