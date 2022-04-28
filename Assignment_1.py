
##### Loading Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.stats import t
from scipy import stats



#### Exercise 1 - Reading and Processing Data 

#### Importing Dataset 'Smoking.txt' 
data = np.loadtxt('smoking.txt')
data

data.shape

age = data[0:,0]
FEV1 = data[0:,1]
height = data[0:,2]
gender = data[0:,3]
smoking_status = data[0:,4]
weight = data[0:,5]



#### Dividing into 'Smokers and Nonsmokers'
smoker = []
nonsmoker = []

smoker = data[np.where(data[:,4] == 1)] 
nonsmoker = data[np.where(data[:,4] == 0)]



#### Average Lung Function (Measured in FEV1) 
FEV1smoker = np.mean(smoker[:,1])
FEV1nonsmoker = np.mean(nonsmoker[:,1])

FEV1smoker
FEV1nonsmoker

# The FEV1 scores for smokers and nonsmokers that I found were 3.2768615384615383 and 2.5661426146010187 respectively. I was suprised to see that the average lung functions were so close also that smokers had a higher lung function than nonsmokers.  




#### Exercise 2 - Boxplots 
#### Boxplot
boxplot1 = [smoker[:,1],nonsmoker [:,1]]
boxplot1

labels = ['Smoker', 'Nonsmoker']
plt.boxplot(boxplot1, labels=labels)
#print(data)
plt.title('Average Lung Capacity - Smokers and Nonsmokers')
plt.ylabel('FEV1 - Liters')
plt.show()

# From the boxplot you can see that the smokers have a higher average lung function than the nonsmokers, but that there were some outliers from the nonsmoking group, as well as more even quartiles and larger whiskers. 




#### Exercise 3 - Hypothesis Testing - Two Sided T Test
#### Variables
###### Mean Smokers / Nonsmokers
FEV1smoker 
FEV1nonsmoker

##### Variance Smokers / Nonsmokers
variancesmoker = np.var(smoker[:,1])
variancenonsmoker = np.var(nonsmoker[:,1])

variancesmoker
variancenonsmoker

###### Number of Smokers / Nonsmokers
smoker.shape[0]
nonsmoker.shape[0]

###### Calculating T Statistic
t_numerator = (FEV1smoker - FEV1nonsmoker)
t_denominator = (variancesmoker/(smoker.shape[0]-1)) + (variancenonsmoker/(nonsmoker.shape[0]-1))
t_denominatorsqrt = np.sqrt(t_denominator)
t_statistic = t_numerator/t_denominatorsqrt
t_statistic

###### Degrees of Freedom - 83
numerator = ((variancesmoker/smoker.shape[0]) + (variancenonsmoker/nonsmoker.shape[0]))**2
denominator1 = ((variancesmoker**2)/(((smoker.shape[0])**2) * (smoker.shape[0]-1)))
denominator2 = ((variancenonsmoker**2)/(((nonsmoker.shape[0])**2) * (nonsmoker.shape[0]-2)))
degrees = numerator / (denominator1 + denominator2) 
degrees
rounded_degrees = np.floor(degrees)
rounded_degrees

###### Calculating P Value
p_value = 2*t.cdf(-t_statistic , rounded_degrees) 
p_value

##### Binary Outcome
if p_value >= 0.05:
    print("Null Hypothesis is true, two group means are equal.")
else:
    print("Null Hypothesis is false, two group means are not equal.")

#I am not surprised by the outcome that the null hypothesis is false, we could tell initially that the two group means were not equal. By doing this T-test however, we determined that not only are the sample means unequal, but that the difference is statistically significant as out p-value is lower than the significance level of 0.05.  




####  Exercise 4 - Correlation
#### Correlation Plot
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(smoker[:,0], smoker[:,1] , label = 'Smoker', color='red')
ax1.scatter(nonsmoker[:,0], nonsmoker[:,1], label = 'Nonsmoker', color='blue')

plt.legend()
plt.title('Average Lung Capacity - Smokers and Nonsmokers')
plt.ylabel('FEV1 - Liters')
plt.xlabel('Age')
plt.show()

#### Correlation value
correlation_value = np.corrcoef(age, FEV1)[0,1]
correlation_value

# We can see that with nonsmokers, lung function generally increases with age. With smokers however, there seems to be much less of a distinguishable correlation, perhaps due to the difference in smoking habits amongst the samples. The correlation found was 0.7564589899895996. 




####  Exercise 5 - Histograms

plt.hist(nonsmoker[:,0],label = 'Nonsmoker', color = 'blue')
plt.hist(smoker[:,0], label = 'Smoker', color = 'red')
plt.xlabel('Age')
plt.ylabel('Total Number')
plt.title('Histogram of Age between Smokers and Nonsmokers within a Test Group')
plt.legend()
plt.show()

# We can see from the histograms that there are very few young smokers, and that the majority of smokers are 12 years old and older, while the nonsmoker samples are in high numbers starting at the age of 6. Because the smoker population is consisting of mostly older members than the majority of the nonsmokers, their average lung function should be higher based on what we know about lung function increasing with age. In this dataset, we see that perhaps the number of younger members of the nonsmoker group might slightly skew the average lung function not because of the smoking status but because of the age. 
