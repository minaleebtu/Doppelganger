import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import cov
from Part2.part2_task2 import numberOfWordsPerComm,  largeWordCountList, simpsonList, sichelList, sentenceLenPerCommList, puncCountPerCommList, multiSpacePerCommList, grammarChkPerCommList, upperWordPerCommList, ease_reading, gunning_fog

print("numberOfWordsPerComm: ", numberOfWordsPerComm)
print("largeWordCountList: ", largeWordCountList)
print("simpsonList: ", simpsonList)
print("sichelList: ", sichelList)
print("sentenceLenPerCommList: ", sentenceLenPerCommList)
print("puncCountPerCommList: ", puncCountPerCommList)
print("multiSpacePerCommList: ", multiSpacePerCommList)
print("grammarChkPerCommList: ", grammarChkPerCommList)
print("upperWordPerCommList: ", upperWordPerCommList)
print("ease_reading: ", ease_reading)
print("gunning_fog: ", gunning_fog)

pd.options.display.float_format = '{:.2f}'.format
# df = pd.DataFrame(numberOfWordsPerComm)
df = pd.DataFrame([x for x in zip(numberOfWordsPerComm,  largeWordCountList, simpsonList, sichelList, sentenceLenPerCommList, puncCountPerCommList, multiSpacePerCommList, grammarChkPerCommList, upperWordPerCommList, ease_reading, gunning_fog)], columns=['numberOfWordsPerComm',  'largeWordCountList', 'simpsonList', 'sichelList', 'sentenceLenPerCommList', 'puncCountPerCommList', 'multiSpacePerCommList', 'grammarChkPerCommList', 'upperWordPerCommList', 'ease_reading', 'gunning_fog'])
# target = df.index
print(df)
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
featureMatrix = df.to_numpy().T
# print("type of featureMatrix: ", type(featureMatrix))
# for i in range(len(featureMatrix)):
#     print("featureMatrix(", i, "): ", featureMatrix[i])
# print("Shape of featureMatrix:\n", np.shape(featureMatrix))
covMatrix = np.cov(featureMatrix, bias=True)
print("Covarinace matrix of featureMatrix:\n", covMatrix)

eigenvalue = np.linalg.eigvals(covMatrix)
eigenvalue, eigenvector = np.linalg.eig(covMatrix)
print("eigenvalue:\n", eigenvalue)
print("eigenvector:\n", eigenvector)

scaled_data = preprocessing.scale(featureMatrix)
pca = PCA()
pca_fit = pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
print("Shape of pca_data:\n", np.shape(pca_data))
print("pca_data:\n", pca_data)
print("type of pca_data: ",  type(pca_data))
selected = []
for i in pca_data:
    print("type of i: ", type(i))
    print("i: ", i)
    cnt = 0

    for j in i:
        if cnt >= 1:
            # selected.append(i.tolist())
            print("cntadsf")
        print("j: ", j)
        if j > 0.99:
            cnt += 1
            print("cnt ++")
            print("cnt: ", cnt)
# print("selected: ", list(set(selected)))


# matrix = StandardScaler().fit_transform(df.values)
# print("matrix: ", matrix)
# print("Covarinace matrix of featureMatrix:\n", np.cov(matrix.round()))

# matrix = np.asmatrix(df.values)


# # print(df)
# COV = np.cov(arr)
# # np.set_printoptions(precision=3)
# print(COV)


# eigval, eigvec = np.linalg.eig(COV)
# print(np.cumsum([i*(100/sum(eigval)) for i in eigval]))