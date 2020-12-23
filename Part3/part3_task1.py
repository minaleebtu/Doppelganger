import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
from numpy import cov
from Part2.part2_task2 import numberOfWordsPerComm,  largeWordCountList, simpsonList, sichelList, sentenceLenPerCommList, puncCountPerCommList, multiSpacePerCommList, grammarChkPerCommList, upperWordPerCommList, ease_reading, gunning_fog


pd.options.display.float_format = '{:.2f}'.format
df = pd.DataFrame([x for x in zip(numberOfWordsPerComm,  largeWordCountList, simpsonList, sichelList, sentenceLenPerCommList, puncCountPerCommList, multiSpacePerCommList, grammarChkPerCommList, upperWordPerCommList, ease_reading, gunning_fog)], columns=['total words per comment',  'frequency of large words per comment', 'Simpson', 'Sichel', 'Average sentence length per comment', 'Frequency of used punctuation per comment', 'Frequency of repeated occurrence of whitespace per comment', 'Number of grammar mistakes per comment', 'Uppercase word usage per comment', 'Ease reading for the content', 'Gunning Fog value for the content'])
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
featureMatrix = df.to_numpy().T
covMatrix = cov(featureMatrix, bias=True)
covDF = pd.DataFrame(data=covMatrix, index=['numberOfWordsPerComm',  'largeWordCountList', 'simpsonList', 'sichelList', 'sentenceLenPerCommList', 'puncCountPerCommList', 'multiSpacePerCommList', 'grammarChkPerCommList', 'upperWordPerCommList', 'ease_reading', 'gunning_fog'])
print("===============================================================================")
print("a) Calculate the covariance matrix of a feature matrix")
print("Covarinace matrix of featureMatrix:\n", covMatrix)
# print("cov pandas:\n ", covDF)

print("===============================================================================")
print("b) Calculate eigenvectors and eigenvalues of the covariance matrix")
eigenvalue = np.linalg.eigvals(covMatrix)
eigenvalue, eigenvector = np.linalg.eig(covMatrix)
print("eigenvector:\n", eigenvector)
print("eigenvalue:\n", eigenvalue)

print("===============================================================================")
print("c) PCA")
scaled_data = preprocessing.scale(featureMatrix)
pca = PCA()
pca_fit = pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
print("pca_data:\n", pca_data)
pcaDF = pd.DataFrame(data=pca_data, index=['total words per comment',  'frequency of large words per comment', 'Simpson', 'Sichel', 'Average sentence length per comment', 'Frequency of used punctuation per comment', 'Frequency of repeated occurrence of whitespace per comment', 'Number of grammar mistakes per comment', 'Uppercase word usage per comment', 'Ease reading for the content', 'Gunning Fog value for the content'])
# print("pca pandas:\n", pcaDF)
pcaList = pcaDF.reset_index().values.tolist()
# print("list of pcaDF: ", pcaList)
selected = []
for i in pcaList:
    cnt = 0
    for j in i[1:]:
        if cnt >= 1:
            selected.append(i[0])
        if j > 0.99:
            cnt += 1
print("Selected features: ", list(set(selected)))
