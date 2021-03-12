# Solution to Problem 2
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

# load full_music_data
music_data = pd.read_csv('full_music_data.csv')

music_data['artist_names'].replace(
    {r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
music_data['release_date'].replace(
    {r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
music_data['song_title (censored)'].replace(
    {r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
music_data.dropna(how='all', inplace=True)
music_data.drop_duplicates(inplace=True)
music_data = music_data.drop(
    ['artist_names', 'artists_id', 'release_date', 'song_title (censored)'], axis=1)

# Define the function that takes the specified columns from the input dataset as the training set and the test set


def xattrSelect(x, idxSet):
    xOut = []
    for row in x:
        xOut.append([row[i] for i in idxSet])
    return(xOut)


xList = []  # Constructing a list to hold the set of attributes
# Extract the tag set from music_data and put it in the list
labels = [float(label) for label in music_data.iloc[:, -1].tolist()]
# Extract the names of all attributes in music_data and put them in the list
names = music_data.columns.tolist()
for i in range(len(music_data)):
    xList.append(music_data.iloc[i, 0:-1])

# Divide the original data set into a training set (2/3 of the set) and a test set (1/3 of the set):
indices = range(len(xList))
xListTest = [xList[i] for i in indices if i % 3 == 0]
xListTrain = [xList[i] for i in indices if i % 3 != 0]
labelsTest = [labels[i] for i in indices if i % 3 == 0]
labelsTrain = [labels[i] for i in indices if i % 3 != 0]


attributeList = []            #
index = range(len(xList[1]))
indexSet = set(index)
oosError = []


for i in index:
    attSet = set(attributeList)
    attTrySet = indexSet - attSet
    attTry = [ii for ii in attTrySet]
    errorList = []
    attTemp = []

    for iTry in attTry:
        attTemp = [] + attributeList
        attTemp.append(iTry)

        # Call the attrSelect function to select the specified columns from xListTrain and xListTest to form a temporary training and test set
        xTrainTemp = xattrSelect(xListTrain, attTemp)
        xTestTemp = xattrSelect(xListTest, attTemp)

        # Convert both the training and test sets into array objects
        xTrain = np.array(xTrainTemp)
        yTrain = np.array(labelsTrain)
        xTest = np.array(xTestTemp)
        yTest = np.array(labelsTest)

        # Training linear regression models with scikit package
        wineQModel = linear_model.LinearRegression()
        wineQModel.fit(xTrain, yTrain)

        # Calculating the RMSE on the test set
        rmsError = np.linalg.norm(
            (yTest-wineQModel.predict(xTest)), 2)/sqrt(len(yTest))
        errorList.append(rmsError)
        attTemp = []

    # Select the new index corresponding to the smallest value in the errorList
    iBest = np.argmin(errorList)
    # Add the corresponding attribute index from attTry to attributeList using the new index iBest
    attributeList.append(attTry[iBest])
    # Add the minimum value from the errorList to the oosError list
    oosError.append(errorList[iBest])

print("Out of sample error versus attribute set size")
print(oosError)
print("\n" + "Best attribute indices")
print(attributeList)
namesList = [names[i] for i in attributeList]
print("\n" + "Best attribute names")
print(namesList)

# Plot the image of RMSE versus number of attributes for linear regression models consisting of different number of attributes on the test set
x = range(len(oosError))
plt.plot(x, oosError, 'k')
plt.xlabel('Number of Attributes')
plt.ylabel('Error (RMS)')
plt.show()

# Plot the histogram of the error distribution on the test set for a linear regression model consisting of the optimal number of attributes
indexBest = oosError.index(min(oosError))
attributesBest = attributeList[1:(indexBest+1)]

# Call the xattrSelect function to select the best number of columns from xListTrain and xListTest to form a temporary training and test set

xTrainTemp = xattrSelect(xListTrain, attributesBest)
xTestTemp = xattrSelect(xListTest, attributesBest)
xTrain = np.array(xTrainTemp)
xTest = np.array(xTestTemp)

# Train the model and plot the histogram
wineQModel = linear_model.LinearRegression()
wineQModel.fit(xTrain, yTrain)
errorVector = yTest-wineQModel.predict(xTest)
plt.hist(errorVector)
plt.xlabel("Bin Boundaries")
plt.ylabel("Counts")
plt.show()

# Plot the scatter plot between actual and predicted values
plt.scatter(wineQModel.predict(xTest), yTest, s=100, alpha=0.10)
plt.xlabel('Predicted  Score')
plt.ylabel('Real  Score')
plt.show()
