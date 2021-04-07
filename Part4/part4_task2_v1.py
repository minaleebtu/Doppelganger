from numpy import argmax
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score

# File containing values of Username, Comment, 11 Features, Label of user
new = pd.read_csv('Joined_v2.csv')
new = new.drop(new.columns[new.columns.str.contains('unnamed', case=False)], axis=1)
# pd.options.display.max_columns = None


# Calculating the PCA to reduce number of features
def getPca(dat_array):
    pca = PCA(n_components=0.999)
    DATA_PCA = pca.fit_transform(dat_array)

    return DATA_PCA


# Getting label value of exact number of users as exact number of comments
def getLabel(sortedData):
    usercnt = sortedData['Label'].value_counts().to_dict()
    numOfComm = list(usercnt.values())[0]


    result = pd.DataFrame()
    for userIn in usercnt.keys():
        result = result.append(sortedData[sortedData.Label == userIn][0:numOfComm], ignore_index=True)

    return result['Label'].to_numpy()


# Getting username of exact number of users
def getUserName(sortedData, numOfUser):
    result = pd.DataFrame()

    for userIn in range(0, numOfUser):
        result = result.append(sortedData[sortedData.Label == userIn], ignore_index=True)

    # Removing duplicated usernames
    newResult = list(set(result['username'].to_list()))

    return newResult


def selectData(numOfP, numOfComm):
    numOfUser = int(numOfP / 2)
    numOfComm = int(numOfComm * 2)
    result = pd.DataFrame()

    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn][0:numOfComm],
                               ignore_index=True)

    result = result.replace(np.nan, '', regex=True)

    return result


def splitPseudonym(selectedData):
    selectedData = shuffle(selectedData)
    usercnt = selectedData['username'].value_counts().to_dict()
    numOfComm = list(usercnt.values())[0]
    grouped = selectedData.groupby(selectedData.username)

    splitList = []

    for uc in usercnt.keys():
        ucList = grouped.get_group(uc).values.tolist()

        A = []
        B = []
        for index in range(len(ucList)):
            if len(A) < int(numOfComm / 2):
                ucList[index][3] += 100
                A.append(ucList[index])
            elif len(A) == int(numOfComm / 2):
                ucList[index][3] += 200
                B.append(ucList[index])
            elif len(A) >= int(numOfComm / 2):
                ucList[index][3] += 200
                B.append(ucList[index])
        splitList.append([A, B])

    # Use list comprehension to convert a list of lists to a flat list
    flatList = [item for elem in splitList for item in elem]
    flatList2 = [item for elem in flatList for item in elem]
    split = pd.DataFrame(flatList2, columns=['articleTitle', 'username', 'comment', 'Label', 'total words per comment',
                                             'frequency of large words per comment', 'Simpson', 'Sichels',
                                             'Average sentence length per comment',
                                             'Frequency of used punctuation per comment',
                                             'Frequency of repeated occurrence of whitespace per comment',
                                             'Number of grammar mistakes per comment',
                                             'Uppercase word usage per comment',
                                             'Ease reading for the content', 'Gunning Fog value for the content'])
    split = split.sort_values(by=['Label', 'username'], ignore_index=True)

    return split


def toPCA(splitData):
    return splitData.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1)


def getCombinedProbs(splitData, prob_per_author):
    avg_prob = {}

    userLabel = splitData['Label'].to_numpy()
    encode_to_num = pd.Series(userLabel, userLabel).to_dict()
    allAuthors = list(encode_to_num.keys())

    for i in range(len(allAuthors)):
        for j in range(i + 1, len(allAuthors)):

            result = 0
            avg = (prob_per_author[i][j] + prob_per_author[j][i]) / 2

            if avg in avg_prob.keys():
                avg_prob[avg] += result
            else:
                avg_prob[avg] = result

    return np.fromiter(avg_prob.keys(), dtype=float)


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def trueY(splitData):
    userLabel = splitData['Label'].to_numpy()
    userLabelDict = pd.Series(np.array(splitData['username'].values.tolist()), userLabel).to_dict()
    encode_to_num = pd.Series(userLabel, userLabel).to_dict()
    allAuthors = list(encode_to_num.keys())

    cols = ['Pseudonym 1', 'Pseudonym 2', 'Encode 1', 'Encode 2']

    resultList = []

    for i in range(len(encode_to_num)):
        a = int(allAuthors[i])
        for j in range(i + 1, len(encode_to_num)):
            b = allAuthors[j]

            resultList.append([userLabelDict[encode_to_num[a]], userLabelDict[encode_to_num[b]], encode_to_num[a], encode_to_num[b]])

    result = pd.DataFrame(resultList, columns=cols)
    result['trueY'] = np.where((result['Pseudonym 1'] == result['Pseudonym 2']), 1, 0)

    return result['trueY'].to_numpy()


# Define the classifiers
clf = svm.SVC(kernel='linear', probability=True)

kfold = KFold(n_splits=3, shuffle=True, random_state=1)

# Separating the dependant and independent variable per experiment
X_20_20 = getPca(toPCA(splitPseudonym(selectData(20, 20))))
X_40_20 = getPca(toPCA(splitPseudonym(selectData(40, 20))))
X_60_20 = getPca(toPCA(splitPseudonym(selectData(60, 20))))
X_60_10 = getPca(toPCA(splitPseudonym(selectData(60, 10))))
X_60_30 = getPca(toPCA(splitPseudonym(selectData(60, 30))))

y_20_20 = getLabel(splitPseudonym(selectData(20, 20)))
y_40_20 = getLabel(splitPseudonym(selectData(40, 20)))
y_60_20 = getLabel(splitPseudonym(selectData(60, 20)))
y_60_10 = getLabel(splitPseudonym(selectData(60, 10)))
y_60_30 = getLabel(splitPseudonym(selectData(60, 30)))

trueY_20_20 = trueY(splitPseudonym(selectData(20, 20)))
trueY_40_20 = trueY(splitPseudonym(selectData(40, 20)))
trueY_60_20 = trueY(splitPseudonym(selectData(60, 20)))
trueY_60_10 = trueY(splitPseudonym(selectData(60, 10)))
trueY_60_30 = trueY(splitPseudonym(selectData(60, 30)))

# for train_ix, test_ix in kfold.split(X_60_20, y_60_20):
#     # select rows
#     train_X, test_X = X_60_20[train_ix], X_60_20[test_ix]
#     train_y, test_y = y_60_20[train_ix], y_60_20[test_ix]
#     print("train_X: ", train_X, type(train_X), train_X.shape)
#     print("test_X: ", test_X, type(test_X), test_X.shape)
#     print("train_y: ", train_y, type(train_y), train_y.shape)
#     print("test_y: ", test_y, type(test_y), test_y.shape)

predictions_20_20_hat = cross_val_predict(clf, X_20_20, y_20_20, cv=kfold, method='predict_proba')
predictions_40_20_hat = cross_val_predict(clf, X_40_20, y_40_20, cv=kfold, method='predict_proba')
predictions_60_20_hat = cross_val_predict(clf, X_60_20, y_60_20, cv=kfold, method='predict_proba')
predictions_60_10_hat = cross_val_predict(clf, X_60_10, y_60_10, cv=kfold, method='predict_proba')
predictions_60_30_hat = cross_val_predict(clf, X_60_30, y_60_30, cv=kfold, method='predict_proba')

print("predictions_60_10_hat: ", predictions_60_10_hat.shape)
print("predictions_60_20_hat: ", predictions_60_20_hat.shape)
print("predictions_60_30_hat: ", predictions_60_30_hat.shape)

# keep probabilities for the positive outcome only
# predictions_20_20 = predictions_20_20_hat[:, 1]
# predictions_40_20 = predictions_40_20_hat[:, 1]
# predictions_60_20 = predictions_60_20_hat[:, 1]
# predictions_60_10 = predictions_60_10_hat[:, 1]
# predictions_60_30 = predictions_60_30_hat[:, 1]

combinedProbs_20_20 = getCombinedProbs(splitPseudonym(selectData(20, 20)), predictions_20_20_hat.tolist())
combinedProbs_40_20 = getCombinedProbs(splitPseudonym(selectData(40, 20)), predictions_40_20_hat.tolist())
combinedProbs_60_20 = getCombinedProbs(splitPseudonym(selectData(60, 20)), predictions_60_20_hat.tolist())
combinedProbs_60_10 = getCombinedProbs(splitPseudonym(selectData(60, 10)), predictions_60_10_hat.tolist())
combinedProbs_60_30 = getCombinedProbs(splitPseudonym(selectData(60, 30)), predictions_60_30_hat.tolist())

# trueY_cnt_20_20 = np.count_nonzero(trueY_20_20 == 1)
# trueY_cnt_40_20 = np.count_nonzero(trueY_40_20 == 1)
# trueY_cnt_60_20 = np.count_nonzero(trueY_60_20 == 1)
# trueY_cnt_60_10 = np.count_nonzero(trueY_60_10 == 1)
# trueY_cnt_60_30 = np.count_nonzero(trueY_60_30 == 1)

# define thresholds
thresholds = np.arange(0, 1, 0.001)

# evaluate each threshold
scores_20_20 = [f1_score(trueY_20_20, to_labels(combinedProbs_20_20, t), average='micro') for t in thresholds]
scores_40_20 = [f1_score(trueY_40_20, to_labels(combinedProbs_40_20, t), average='micro') for t in thresholds]
scores_60_20 = [f1_score(trueY_60_20, to_labels(combinedProbs_60_20, t), average='micro') for t in thresholds]
scores_60_10 = [f1_score(trueY_60_10, to_labels(combinedProbs_60_10, t), average='micro') for t in thresholds]
scores_60_30 = [f1_score(trueY_60_30, to_labels(combinedProbs_60_30, t), average='micro') for t in thresholds]

# get best threshold
ix_20_20 = argmax(scores_20_20)
ix_40_20 = argmax(scores_40_20)
ix_60_20 = argmax(scores_60_20)
ix_60_10 = argmax(scores_60_10)
ix_60_30 = argmax(scores_60_30)

print('[20 Pseudonyms & 20 Comments] Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix_20_20], scores_20_20[ix_20_20]))
print('[40 Pseudonyms & 20 Comments] Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix_40_20], scores_40_20[ix_40_20]))
print('[60 Pseudonyms & 20 Comments] Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix_60_20], scores_60_20[ix_60_20]))
print('-'*80)
print('[60 Pseudonyms & 10 Comments] Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix_60_10], scores_60_10[ix_60_10]))
print('[60 Pseudonyms & 20 Comments] Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix_60_20], scores_60_20[ix_60_20]))
print('[60 Pseudonyms & 30 Comments] Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix_60_30], scores_60_30[ix_60_30]))
print('='*80)

doppel_20_20 = np.where(combinedProbs_20_20 > thresholds[ix_20_20], 1, 0)
doppel_40_20 = np.where(combinedProbs_40_20 > thresholds[ix_40_20], 1, 0)
doppel_60_20 = np.where(combinedProbs_60_20 > thresholds[ix_60_20], 1, 0)
doppel_60_10 = np.where(combinedProbs_60_10 > thresholds[ix_60_10], 1, 0)
doppel_60_30 = np.where(combinedProbs_60_30 > thresholds[ix_60_30], 1, 0)

# doppel_cnt_20_20 = np.count_nonzero(doppel_20_20 == 1)
# doppel_cnt_40_20 = np.count_nonzero(doppel_40_20 == 1)
# doppel_cnt_60_20 = np.count_nonzero(doppel_60_20 == 1)
# doppel_cnt_60_10 = np.count_nonzero(doppel_60_10 == 1)
# doppel_cnt_60_30 = np.count_nonzero(doppel_60_30 == 1)

print("Accuracy Score With 20 Pseudonyms & 20 Comments : ", accuracy_score(trueY_20_20, doppel_20_20))
print("Accuracy Score With 40 Pseudonyms & 20 Comments : ", accuracy_score(trueY_40_20, doppel_40_20))
print("Accuracy Score With 60 Pseudonyms & 20 Comments : ", accuracy_score(trueY_60_20, doppel_60_20))
print('-'*80)
print("Accuracy Score With 60 Pseudonyms & 10 Comments : ", accuracy_score(trueY_60_10, doppel_60_10))
print("Accuracy Score With 60 Pseudonyms & 20 Comments : ", accuracy_score(trueY_60_20, doppel_60_20))
print("Accuracy Score With 60 Pseudonyms & 30 Comments : ", accuracy_score(trueY_60_30, doppel_60_30))

x_ticks = ['20 Pseudonyms', '40 Pseudonyms', '60 Pseudonyms', '10 Comments', '20 Comments', '30 Comments']
numOfP = [accuracy_score(trueY_20_20, doppel_20_20), accuracy_score(trueY_40_20, doppel_40_20),
                accuracy_score(trueY_60_20, doppel_60_20)]

numOfCommPerP = [accuracy_score(trueY_60_10, doppel_60_10), accuracy_score(trueY_60_20, doppel_60_20),
                   accuracy_score(trueY_60_30, doppel_60_30)]

x_axis = np.arange(6)

plt.bar(x_axis[0:3], numOfP, color='maroon', label='With 20 Comments')
plt.bar(x_axis[3:6], numOfCommPerP, color='salmon', label='per 60 Pseudonyms')
plt.xticks(x_axis, x_ticks, fontsize=6)

plt.legend()
plt.xlabel('Experiments')
plt.ylabel('Accuracy score')
plt.show()
