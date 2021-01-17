import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import numpy as np

new = pd.read_csv('Joinedv1.csv')
datdrop = new.drop(['Unnamed: 0', 'username', 'comment', 'Label'], axis = 1)

# Convert the dataset into an array
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
dat_array = datdrop.to_numpy()

def getPca(dat_array):
    # Calculate the PCA and extract the features
    pd.options.display.width = 0
    pca = PCA(n_components = 0.999)
    DATA_PCA = pca.fit_transform(dat_array)
    # print("PCA Data : \n", DATA_PCA, DATA_PCA.shape)
    return DATA_PCA


def getLabel(numOfUser, numOfComm):
    result = pd.DataFrame()
    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn][0:numOfComm], ignore_index=True)
    newResult = result['Label'].to_numpy()
    return newResult


def selectData(numOfUser, numOfComm):
    result = pd.DataFrame()
    for userIn in range(0, numOfUser):
        # print("new[new.Label == ", userIn, "]\n", new[new.Label == userIn].drop(['Unnamed: 0', 'comment'], axis = 1)[0:numOfComm], "\n", len(new[new.Label == userIn].drop(['Unnamed: 0', 'comment'], axis = 1)[0:numOfComm]))
        result = result.append(new[new.Label == userIn].drop(['Unnamed: 0', 'comment'], axis=1)[0:numOfComm], ignore_index = True)
    result = result.drop(['username', 'Label'], axis=1)

    return result.to_numpy()

# print("selectData: ", selectData(20, 20))

print("getPca:\n", getPca(selectData(40, 20)), getPca(selectData(40, 20)).shape)
# print("getLabel: ", getLabel(20, 20), getLabel(20, 20).shape)


X = getPca(selectData(20, 20))
y = getLabel(20, 20)

clf = svm.SVC(kernel='linear')
cross = cross_val_score(clf, X, y, cv=3)
print("cross_val_score: ", cross)
