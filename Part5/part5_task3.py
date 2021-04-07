import os
import random
import timeit
import tracemalloc

import psutil
from numpy import argmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from features import getFeatures
from sklearn.metrics import f1_score, accuracy_score

# File containing values of Username, Comment, 11 Features, Label of user
new = pd.read_csv('Joined_v2.csv')
new = new.drop(new.columns[new.columns.str.contains('unnamed', case=False)], axis=1)

kfold = GroupKFold(n_splits=3)


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


def selectData(numOfP, numOfComm):
    numOfUser = int(numOfP / 2)
    numOfComm = int(numOfComm * 2)
    result = pd.DataFrame()

    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn][0:numOfComm],
                               ignore_index=True)

    result = result.replace(np.nan, '', regex=True)

    return result


def splitAll(selectedData):
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
    # 4D to 3D
    flatList = [item for elem in splitList for item in elem]
    # 3D to 2D
    flatList2 = [item for elem in flatList for item in elem]
    split = pd.DataFrame(flatList2, columns=['articleTitle', 'username', 'comment', 'Label', 'total words per comment',
                                      'frequency of large words per comment', 'Simpson', 'Sichels',
                                      'Average sentence length per comment',
                                      'Frequency of used punctuation per comment',
                                      'Frequency of repeated occurrence of whitespace per comment',
                                      'Number of grammar mistakes per comment', 'Uppercase word usage per comment',
                                      'Ease reading for the content', 'Gunning Fog value for the content'])
    split = split.sort_values(by=['Label', 'username'], ignore_index=True)

    return split


def splitSingle(selectedData):
    selectedData = shuffle(selectedData)
    usercnt = selectedData['username'].value_counts().to_dict()
    randomOne = random.choice(list(usercnt.keys()))
    numOfComm = usercnt[randomOne]

    del usercnt[randomOne]

    grouped = selectedData.groupby(selectedData.username)

    splitList = []
    ucList = grouped.get_group(randomOne).values.tolist()

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
    # 4D to 3D
    flatList = [item for elem in splitList for item in elem]
    # 3D to 2D
    flatList2 = [item for elem in flatList for item in elem]
    split = pd.DataFrame(flatList2, columns=['articleTitle', 'username', 'comment', 'Label', 'total words per comment',
                                      'frequency of large words per comment', 'Simpson', 'Sichels',
                                      'Average sentence length per comment',
                                      'Frequency of used punctuation per comment',
                                      'Frequency of repeated occurrence of whitespace per comment',
                                      'Number of grammar mistakes per comment', 'Uppercase word usage per comment',
                                      'Ease reading for the content', 'Gunning Fog value for the content'])
    split = split.append(selectedData[~selectedData.username.str.contains(randomOne)], ignore_index=True)
    split = split.sort_values(by=['Label', 'username'], ignore_index=True)

    return split


def splitRandom(selectedData):
    percent = float("{:.2f}".format(random.uniform(0.25, 0.75)))
    selectedData = shuffle(selectedData)
    usercnt = selectedData['username'].value_counts().to_dict()
    numOfUser = int(len(usercnt) * percent)

    randomP = random.sample(list(usercnt.keys()), numOfUser)
    numOfComm = list(usercnt.values())[0]

    grouped = selectedData.groupby(selectedData.username)

    splitList = []
    for uc in randomP:
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
    # 4D to 3D
    flatList = [item for elem in splitList for item in elem]
    # 3D to 2D
    flatList2 = [item for elem in flatList for item in elem]
    split = pd.DataFrame(flatList2, columns=['articleTitle', 'username', 'comment', 'Label', 'total words per comment',
                                      'frequency of large words per comment', 'Simpson', 'Sichels',
                                      'Average sentence length per comment',
                                      'Frequency of used punctuation per comment',
                                      'Frequency of repeated occurrence of whitespace per comment',
                                      'Number of grammar mistakes per comment', 'Uppercase word usage per comment',
                                      'Ease reading for the content', 'Gunning Fog value for the content'])
    split = split.append(selectedData[~selectedData.username.str.contains('|'.join(randomP))], ignore_index=True)
    split = split.sort_values(by=['Label', 'username'], ignore_index=True)

    return split


def getCombinedProbs(y_arr, prob_per_author):
    avg_prob = []

    y_arr_list = list(set(y_arr.tolist()))

    for i in range(len(y_arr_list)):
        for j in range(i + 1, len(y_arr_list)):
            avg = (prob_per_author[i][j] + prob_per_author[j][i]) / 2

            avg_prob.append(avg)

    return np.array(avg_prob)


def trueY_label(y_arr):
    y_arr_list = list(set(y_arr.tolist()))
    cnt = 0
    for en in y_arr_list:
        if len(str(en)) >= 3:
            cnt += 1
    if cnt == 0:
        return np.zeros(int((len(y_arr_list) * (len(y_arr_list) - 1)) / 2))

    resultList = []
    for i in range(len(y_arr_list)):
        for j in range(i + 1, len(y_arr_list)):
            if str(y_arr_list[i])[1:] == str(y_arr_list[j])[1:]:
                resultList.append(1)
            else:
                resultList.append(0)
    return np.array(resultList)


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def quota_same(scenario, numOfP, numOfComm, clfName):
    split_none = selectData(numOfP, numOfComm)
    split_all = splitAll(selectData(numOfP, numOfComm))

    if scenario == 'none':
        X = getPca(split_none.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_none)
    elif scenario == 'all':
        X = getPca(split_all.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_all)

    if clfName.upper() == 'SVM':
        clf = svm.SVC(kernel='linear', probability=True)
    elif clfName.upper() == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif clfName.upper() == 'RF':
        clf = RandomForestClassifier(n_estimators=40)

    fold_cnt = 0
    acc_score = []
    # enumerate the splits
    for train_ix, test_ix in kfold.split(X, y, groups=y):
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)
        yhat = cross_val_predict(clf, train_X, train_y, cv=kfold, method='predict_proba', groups=train_y)
        combinedProbs = getCombinedProbs(train_y, yhat.tolist())

        # define thresholds
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [f1_score(trueY_label(train_y), to_labels(combinedProbs, t), average='micro') for t in thresholds]
        ix = argmax(scores)
        fold_cnt += 1
        print('[Quota Same/', scenario, '/', numOfP, 'Pseudonyms &', numOfComm, 'Comments/', fold_cnt,
              'fold] Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
        yhat_test = cross_val_predict(clf, test_X, test_y, cv=kfold, method='predict_proba', groups=test_y)
        y_pred_combined = (getCombinedProbs(test_y, yhat_test.tolist()) >= thresholds[ix]).astype('int')
        acc = accuracy_score(trueY_label(test_y), y_pred_combined)
        acc_score.append(acc)
    print('-' * 80)
    avg_acc_score = sum(acc_score) / kfold.get_n_splits()

    return avg_acc_score


def quota_exclude(scenario, numOfP, numOfComm, clfName):
    split_none = selectData(numOfP, numOfComm)
    split_single = splitSingle(selectData(numOfP, numOfComm))
    split_random = splitRandom(selectData(numOfP, numOfComm))
    split_all = splitAll(selectData(numOfP, numOfComm))

    appended_dataset = pd.DataFrame()
    if scenario == 'none':
        appended_dataset = appended_dataset.append([split_single, split_random, split_all])
        X = getPca(split_none.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_none)
    elif scenario == 'single':
        appended_dataset = appended_dataset.append([split_none, split_random, split_all], ignore_index=True)
        X = getPca(split_single.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_single)
    elif scenario == 'random':
        appended_dataset = appended_dataset.append([split_none, split_single, split_all], ignore_index=True)
        X = getPca(split_random.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_random)
    elif scenario == 'all':
        appended_dataset = appended_dataset.append([split_none, split_single, split_random], ignore_index=True)
        X = getPca(split_all.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_all)

    if clfName.upper() == 'SVM':
        clf = svm.SVC(kernel='linear', probability=True)
    elif clfName.upper() == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif clfName.upper() == 'RF':
        clf = RandomForestClassifier(n_estimators=40)

    appended_PCA = appended_dataset.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1)

    appended_X = getPca(appended_PCA)
    appended_y = getLabel(appended_dataset)

    fold_cnt = 0
    acc_score = []
    # enumerate the splits
    for train_ix, test_ix in kfold.split(X, y, groups=y):
        train_X, train_y = appended_X[train_ix], appended_y[train_ix]
        test_X, test_y = X[test_ix], y[test_ix]
        clf.fit(train_X, train_y)
        yhat = cross_val_predict(clf, train_X, train_y, cv=kfold, method='predict_proba', groups=train_y)
        combinedProbs_train = getCombinedProbs(train_y, yhat.tolist())

        # define thresholds
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [f1_score(trueY_label(train_y), to_labels(combinedProbs_train, t), average='micro') for t in thresholds]
        ix = argmax(scores)
        fold_cnt += 1
        print('[Quota Exclude /', scenario, '/', numOfP, 'Pseudonyms &', numOfComm, 'Comments/', fold_cnt,
              'fold] Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

        yhat_test = cross_val_predict(clf, test_X, test_y, cv=kfold, method='predict_proba', groups=test_y)
        y_pred = (getCombinedProbs(test_y, yhat_test.tolist()) >= thresholds[ix]).astype('int')
        acc = accuracy_score(trueY_label(test_y), y_pred)
        acc_score.append(acc)
    print('-' * 80)
    avg_acc_score = sum(acc_score) / kfold.get_n_splits()

    return avg_acc_score


def quota_include(scenario, numOfP, numOfComm, clfName):
    split_none = selectData(numOfP, numOfComm)
    split_single = splitSingle(selectData(numOfP, numOfComm))
    split_random = splitRandom(selectData(numOfP, numOfComm))
    split_all = splitAll(selectData(numOfP, numOfComm))

    appended_dataset = pd.DataFrame()
    appended_dataset = appended_dataset.append([split_none, split_single, split_random, split_all])
    if scenario == 'none':
        X = getPca(split_none.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_none)
    elif scenario == 'single':
        X = getPca(split_single.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_single)
    elif scenario == 'random':
        X = getPca(split_random.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_random)
    elif scenario == 'all':
        X = getPca(split_all.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_all)

    if clfName.upper() == 'SVM':
        clf = svm.SVC(kernel='linear', probability=True)
    elif clfName.upper() == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=3)
    elif clfName.upper() == 'RF':
        clf = RandomForestClassifier(n_estimators=40)

    appended_PCA = appended_dataset.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1)

    appended_X = getPca(appended_PCA)
    appended_y = getLabel(appended_dataset)

    fold_cnt = 0
    acc_score = []
    # enumerate the splits
    for train_ix, test_ix in kfold.split(X, y, groups=y):
        train_X, train_y = appended_X[train_ix], appended_y[train_ix]
        test_X, test_y = X[test_ix], y[test_ix]
        clf.fit(train_X, train_y)
        yhat = cross_val_predict(clf, train_X, train_y, cv=kfold, method='predict_proba', groups=train_y)
        combinedProbs_train = getCombinedProbs(train_y, yhat.tolist())

        # define thresholds
        thresholds = np.arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [f1_score(trueY_label(train_y), to_labels(combinedProbs_train, t), average='micro') for t in thresholds]
        ix = argmax(scores)
        fold_cnt += 1
        print('[Quota Include /', scenario, '/', numOfP, 'Pseudonyms &', numOfComm, 'Comments/', fold_cnt,
              'fold] Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

        yhat_test = cross_val_predict(clf, test_X, test_y, cv=kfold, method='predict_proba', groups=test_y)
        y_pred = (getCombinedProbs(test_y, yhat_test.tolist()) >= thresholds[ix]).astype('int')
        acc = accuracy_score(trueY_label(test_y), y_pred)
        acc_score.append(acc)
    print('-' * 80)
    avg_acc_score = sum(acc_score) / kfold.get_n_splits()

    return avg_acc_score


def memUsage(function):
    # tracemalloc.start()
    function
    # mem_usage = tracemalloc.get_tracemalloc_memory()
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"[{function}]Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # tracemalloc.stop()
    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f"Memory Usage is {mem_usage}MB")
    print("--" * 30)
    # tracemalloc.clear_traces()
    # tracemalloc.stop()

    return mem_usage

    # return current / 10**6
    # pid = os.getpid()
    # current_process = psutil.Process(pid)
    # current_process_memory_usage_as_MB = current_process.memory_info()[0] / 10**6
    # print(f"Current memory MB   : {current_process_memory_usage_as_MB:9.3f}MB")
    # print("--" * 30)

    # return current_process_memory_usage_as_MB


# Runtime for feature extraction a)
run_features_20_20 = timeit.timeit('getFeatures(splitAll(selectData(20, 20)))',
                                   setup='from __main__ import getFeatures, splitAll, selectData', number=1)
run_features_50_20 = timeit.timeit('getFeatures(splitAll(selectData(50, 20)))',
                                   setup='from __main__ import getFeatures, splitAll, selectData', number=1)
run_features_70_20 = timeit.timeit('getFeatures(splitAll(selectData(70, 20)))',
                                   setup='from __main__ import getFeatures, splitAll, selectData', number=1)

# Runtime for feature extraction b)
run_features_100_10 = timeit.timeit('getFeatures(splitAll(selectData(100, 10)))',
                                   setup='from __main__ import getFeatures, splitAll, selectData', number=1)
run_features_100_15 = timeit.timeit('getFeatures(splitAll(selectData(100, 15)))',
                                   setup='from __main__ import getFeatures, splitAll, selectData', number=1)
run_features_100_20 = timeit.timeit('getFeatures(splitAll(selectData(100, 20)))',
                                   setup='from __main__ import getFeatures, splitAll, selectData', number=1)
run_features_100_25 = timeit.timeit('getFeatures(splitAll(selectData(100, 25)))',
                                   setup='from __main__ import getFeatures, splitAll, selectData', number=1)
# =========================================================================================

# Memory for feature extraction a)
mem_features_20_20 = memUsage(getFeatures(splitAll(selectData(20, 20))))
mem_features_50_20 = memUsage(getFeatures(splitAll(selectData(50, 20))))
mem_features_70_20 = memUsage(getFeatures(splitAll(selectData(70, 20))))

# Memory for feature extraction b)
mem_features_100_10 = memUsage(getFeatures(splitAll(selectData(100, 10))))
mem_features_100_15 = memUsage(getFeatures(splitAll(selectData(100, 15))))
mem_features_100_20 = memUsage(getFeatures(splitAll(selectData(100, 20))))
mem_features_100_25 = memUsage(getFeatures(splitAll(selectData(100, 25))))
# =========================================================================================
run_features_a = [run_features_20_20, run_features_50_20, run_features_70_20, run_features_100_20]
run_features_b = [run_features_100_10, run_features_100_15, run_features_100_20, run_features_100_25]

mem_features_a = [mem_features_20_20, mem_features_50_20, mem_features_70_20, mem_features_100_20]
mem_features_b = [mem_features_100_10, mem_features_100_15, mem_features_100_20, mem_features_100_25]
# =========================================================================================

# Runtime for doppel SVM a)
run_doppel_same_20_20_svm = timeit.timeit('quota_same("all", 20, 20, "svm")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_50_20_svm = timeit.timeit('quota_same("all", 50, 20, "svm")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_70_20_svm = timeit.timeit('quota_same("all", 70, 20, "svm")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_20_svm = timeit.timeit('quota_same("all", 100, 20, "svm")',
                                          setup='from __main__ import quota_same', number=1)

run_doppel_ex_20_20_svm = timeit.timeit('quota_exclude("all", 20, 20, "svm")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_50_20_svm = timeit.timeit('quota_exclude("all", 50, 20, "svm")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_70_20_svm = timeit.timeit('quota_exclude("all", 70, 20, "svm")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_20_svm = timeit.timeit('quota_exclude("all", 100, 20, "svm")',
                                          setup='from __main__ import quota_exclude', number=1)

run_doppel_in_20_20_svm = timeit.timeit('quota_include("all", 20, 20, "svm")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_50_20_svm = timeit.timeit('quota_include("all", 50, 20, "svm")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_70_20_svm = timeit.timeit('quota_include("all", 70, 20, "svm")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_20_svm = timeit.timeit('quota_include("all", 100, 20, "svm")',
                                          setup='from __main__ import quota_include', number=1)
# =========================================================================================
run_doppel_same_svm_a = [run_doppel_same_20_20_svm, run_doppel_same_50_20_svm, run_doppel_same_70_20_svm, run_doppel_same_100_20_svm]
run_doppel_ex_svm_a = [run_doppel_ex_20_20_svm, run_doppel_ex_50_20_svm, run_doppel_ex_70_20_svm, run_doppel_ex_100_20_svm]
run_doppel_in_svm_a = [run_doppel_in_20_20_svm, run_doppel_in_50_20_svm, run_doppel_in_70_20_svm, run_doppel_in_100_20_svm]
# =========================================================================================

# Runtime for doppel SVM b)
run_doppel_same_100_10_svm = timeit.timeit('quota_same("all", 100, 10, "svm")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_15_svm = timeit.timeit('quota_same("all", 100, 15, "svm")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_25_svm = timeit.timeit('quota_same("all", 100, 25, "svm")',
                                          setup='from __main__ import quota_same', number=1)

run_doppel_ex_100_10_svm = timeit.timeit('quota_exclude("all", 100, 10, "svm")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_15_svm = timeit.timeit('quota_exclude("all", 100, 15, "svm")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_25_svm = timeit.timeit('quota_exclude("all", 100, 25, "svm")',
                                          setup='from __main__ import quota_exclude', number=1)

run_doppel_in_100_10_svm = timeit.timeit('quota_include("all", 100, 10, "svm")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_15_svm = timeit.timeit('quota_include("all", 100, 15, "svm")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_25_svm = timeit.timeit('quota_include("all", 100, 25, "svm")',
                                          setup='from __main__ import quota_include', number=1)
# =========================================================================================
run_doppel_same_svm_b = [run_doppel_same_100_10_svm, run_doppel_same_100_15_svm, run_doppel_same_100_20_svm, run_doppel_same_100_25_svm]
run_doppel_ex_svm_b = [run_doppel_ex_100_10_svm, run_doppel_ex_100_15_svm, run_doppel_ex_100_20_svm, run_doppel_ex_100_25_svm]
run_doppel_in_svm_b = [run_doppel_in_100_10_svm, run_doppel_in_100_15_svm, run_doppel_in_100_20_svm, run_doppel_in_100_25_svm]
# =========================================================================================

# Runtime for doppel KNN a)
run_doppel_same_20_20_knn = timeit.timeit('quota_same("all", 20, 20, "knn")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_50_20_knn = timeit.timeit('quota_same("all", 50, 20, "knn")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_70_20_knn = timeit.timeit('quota_same("all", 70, 20, "knn")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_20_knn = timeit.timeit('quota_same("all", 100, 20, "knn")',
                                          setup='from __main__ import quota_same', number=1)

run_doppel_ex_20_20_knn = timeit.timeit('quota_exclude("all", 20, 20, "knn")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_50_20_knn = timeit.timeit('quota_exclude("all", 50, 20, "knn")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_70_20_knn = timeit.timeit('quota_exclude("all", 70, 20, "knn")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_20_knn = timeit.timeit('quota_exclude("all", 100, 20, "knn")',
                                          setup='from __main__ import quota_exclude', number=1)

run_doppel_in_20_20_knn = timeit.timeit('quota_include("all", 20, 20, "knn")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_50_20_knn = timeit.timeit('quota_include("all", 50, 20, "knn")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_70_20_knn = timeit.timeit('quota_include("all", 70, 20, "knn")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_20_knn = timeit.timeit('quota_include("all", 100, 20, "knn")',
                                          setup='from __main__ import quota_include', number=1)
# =========================================================================================
run_doppel_same_knn_a = [run_doppel_same_20_20_knn, run_doppel_same_50_20_knn, run_doppel_same_70_20_knn, run_doppel_same_100_20_knn]
run_doppel_ex_knn_a = [run_doppel_ex_20_20_knn, run_doppel_ex_50_20_knn, run_doppel_ex_70_20_knn, run_doppel_ex_100_20_knn]
run_doppel_in_knn_a = [run_doppel_in_20_20_knn, run_doppel_in_50_20_knn, run_doppel_in_70_20_knn, run_doppel_in_100_20_knn]
# =========================================================================================

# Runtime for doppel KNN b)
run_doppel_same_100_10_knn = timeit.timeit('quota_same("all", 100, 10, "knn")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_15_knn = timeit.timeit('quota_same("all", 100, 15, "knn")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_25_knn = timeit.timeit('quota_same("all", 100, 25, "knn")',
                                          setup='from __main__ import quota_same', number=1)

run_doppel_ex_100_10_knn = timeit.timeit('quota_exclude("all", 100, 10, "knn")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_15_knn = timeit.timeit('quota_exclude("all", 100, 15, "knn")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_25_knn = timeit.timeit('quota_exclude("all", 100, 25, "knn")',
                                          setup='from __main__ import quota_exclude', number=1)

run_doppel_in_100_10_knn = timeit.timeit('quota_include("all", 100, 10, "knn")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_15_knn = timeit.timeit('quota_include("all", 100, 15, "knn")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_25_knn = timeit.timeit('quota_include("all", 100, 25, "knn")',
                                          setup='from __main__ import quota_include', number=1)
# =========================================================================================
run_doppel_same_knn_b = [run_doppel_same_100_10_knn, run_doppel_same_100_15_knn, run_doppel_same_100_20_knn, run_doppel_same_100_25_knn]
run_doppel_ex_knn_b = [run_doppel_ex_100_10_knn, run_doppel_ex_100_15_knn, run_doppel_ex_100_20_knn, run_doppel_ex_100_25_knn]
run_doppel_in_knn_b = [run_doppel_in_100_10_knn, run_doppel_in_100_15_knn, run_doppel_in_100_20_knn, run_doppel_in_100_25_knn]
# =========================================================================================

# Runtime for doppel RF a)
run_doppel_same_20_20_rf = timeit.timeit('quota_same("all", 20, 20, "rf")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_50_20_rf = timeit.timeit('quota_same("all", 50, 20, "rf")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_70_20_rf = timeit.timeit('quota_same("all", 70, 20, "rf")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_20_rf = timeit.timeit('quota_same("all", 100, 20, "rf")',
                                          setup='from __main__ import quota_same', number=1)

run_doppel_ex_20_20_rf = timeit.timeit('quota_exclude("all", 20, 20, "rf")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_50_20_rf = timeit.timeit('quota_exclude("all", 50, 20, "rf")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_70_20_rf = timeit.timeit('quota_exclude("all", 70, 20, "rf")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_20_rf = timeit.timeit('quota_exclude("all", 100, 20, "rf")',
                                          setup='from __main__ import quota_exclude', number=1)

run_doppel_in_20_20_rf = timeit.timeit('quota_include("all", 20, 20, "rf")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_50_20_rf = timeit.timeit('quota_include("all", 50, 20, "rf")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_70_20_rf = timeit.timeit('quota_include("all", 70, 20, "rf")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_20_rf = timeit.timeit('quota_include("all", 100, 20, "rf")',
                                          setup='from __main__ import quota_include', number=1)
# =========================================================================================
run_doppel_same_rf_a = [run_doppel_same_20_20_rf, run_doppel_same_50_20_rf, run_doppel_same_70_20_rf, run_doppel_same_100_20_rf]
run_doppel_ex_rf_a = [run_doppel_ex_20_20_rf, run_doppel_ex_50_20_rf, run_doppel_ex_70_20_rf, run_doppel_ex_100_20_rf]
run_doppel_in_rf_a = [run_doppel_in_20_20_rf, run_doppel_in_50_20_rf, run_doppel_in_70_20_rf, run_doppel_in_100_20_rf]
# =========================================================================================

# Runtime for doppel RF b)
run_doppel_same_100_10_rf = timeit.timeit('quota_same("all", 100, 10, "rf")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_15_rf = timeit.timeit('quota_same("all", 100, 15, "rf")',
                                          setup='from __main__ import quota_same', number=1)
run_doppel_same_100_25_rf = timeit.timeit('quota_same("all", 100, 25, "rf")',
                                          setup='from __main__ import quota_same', number=1)

run_doppel_ex_100_10_rf = timeit.timeit('quota_exclude("all", 100, 10, "rf")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_15_rf = timeit.timeit('quota_exclude("all", 100, 15, "rf")',
                                          setup='from __main__ import quota_exclude', number=1)
run_doppel_ex_100_25_rf = timeit.timeit('quota_exclude("all", 100, 25, "rf")',
                                          setup='from __main__ import quota_exclude', number=1)

run_doppel_in_100_10_rf = timeit.timeit('quota_include("all", 100, 10, "rf")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_15_rf = timeit.timeit('quota_include("all", 100, 15, "rf")',
                                          setup='from __main__ import quota_include', number=1)
run_doppel_in_100_25_rf = timeit.timeit('quota_include("all", 100, 25, "rf")',
                                          setup='from __main__ import quota_include', number=1)
# =========================================================================================
run_doppel_same_rf_b = [run_doppel_same_100_10_rf, run_doppel_same_100_15_rf, run_doppel_same_100_20_rf, run_doppel_same_100_25_rf]
run_doppel_ex_rf_b = [run_doppel_ex_100_10_rf, run_doppel_ex_100_15_rf, run_doppel_ex_100_20_rf, run_doppel_ex_100_25_rf]
run_doppel_in_rf_b = [run_doppel_in_100_10_rf, run_doppel_in_100_15_rf, run_doppel_in_100_20_rf, run_doppel_in_100_25_rf]
# =========================================================================================

# Memory for doppel SVM a)
mem_doppel_same_20_20_svm = memUsage(quota_same("all", 20, 20, "svm"))
mem_doppel_same_50_20_svm = memUsage(quota_same("all", 50, 20, "svm"))
mem_doppel_same_70_20_svm = memUsage(quota_same("all", 70, 20, "svm"))
mem_doppel_same_100_20_svm = memUsage(quota_same("all", 100, 20, "svm"))

mem_doppel_ex_20_20_svm = memUsage(quota_exclude("all", 20, 20, "svm"))
mem_doppel_ex_50_20_svm = memUsage(quota_exclude("all", 50, 20, "svm"))
mem_doppel_ex_70_20_svm = memUsage(quota_exclude("all", 70, 20, "svm"))
mem_doppel_ex_100_20_svm = memUsage(quota_exclude("all", 100, 20, "svm"))

mem_doppel_in_20_20_svm = memUsage(quota_include("all", 20, 20, "svm"))
mem_doppel_in_50_20_svm = memUsage(quota_include("all", 50, 20, "svm"))
mem_doppel_in_70_20_svm = memUsage(quota_include("all", 70, 20, "svm"))
mem_doppel_in_100_20_svm = memUsage(quota_include("all", 100, 20, "svm"))
# =========================================================================================
mem_doppel_same_svm_a = [mem_doppel_same_20_20_svm, mem_doppel_same_50_20_svm, mem_doppel_same_70_20_svm, mem_doppel_same_100_20_svm]
mem_doppel_ex_svm_a = [mem_doppel_ex_20_20_svm, mem_doppel_ex_50_20_svm, mem_doppel_ex_70_20_svm, mem_doppel_ex_100_20_svm]
mem_doppel_in_svm_a = [mem_doppel_in_20_20_svm, mem_doppel_in_50_20_svm, mem_doppel_in_70_20_svm, mem_doppel_in_100_20_svm]
# =========================================================================================

# Memory for doppel SVM b)
mem_doppel_same_100_10_svm = memUsage(quota_same("all", 100, 10, "svm"))
mem_doppel_same_100_15_svm = memUsage(quota_same("all", 100, 15, "svm"))
mem_doppel_same_100_25_svm = memUsage(quota_same("all", 100, 25, "svm"))

mem_doppel_ex_100_10_svm = memUsage(quota_exclude("all", 100, 10, "svm"))
mem_doppel_ex_100_15_svm = memUsage(quota_exclude("all", 100, 15, "svm"))
mem_doppel_ex_100_25_svm = memUsage(quota_exclude("all", 100, 25, "svm"))

mem_doppel_in_100_10_svm = memUsage(quota_include("all", 100, 10, "svm"))
mem_doppel_in_100_15_svm = memUsage(quota_include("all", 100, 15, "svm"))
mem_doppel_in_100_25_svm = memUsage(quota_include("all", 100, 25, "svm"))
# =========================================================================================
mem_doppel_same_svm_b = [mem_doppel_same_100_10_svm, mem_doppel_same_100_15_svm, mem_doppel_same_100_20_svm, mem_doppel_same_100_25_svm]
mem_doppel_ex_svm_b = [mem_doppel_ex_100_10_svm, mem_doppel_ex_100_15_svm, mem_doppel_ex_100_20_svm, mem_doppel_ex_100_25_svm]
mem_doppel_in_svm_b = [mem_doppel_in_100_10_svm, mem_doppel_in_100_15_svm, mem_doppel_in_100_20_svm, mem_doppel_in_100_25_svm]
# =========================================================================================

# Memory for doppel KNN a)
mem_doppel_same_20_20_knn = memUsage(quota_same("all", 20, 20, "knn"))
mem_doppel_same_50_20_knn = memUsage(quota_same("all", 50, 20, "knn"))
mem_doppel_same_70_20_knn = memUsage(quota_same("all", 70, 20, "knn"))
mem_doppel_same_100_20_knn = memUsage(quota_same("all", 100, 20, "knn"))

mem_doppel_ex_20_20_knn = memUsage(quota_exclude("all", 20, 20, "knn"))
mem_doppel_ex_50_20_knn = memUsage(quota_exclude("all", 50, 20, "knn"))
mem_doppel_ex_70_20_knn = memUsage(quota_exclude("all", 70, 20, "knn"))
mem_doppel_ex_100_20_knn = memUsage(quota_exclude("all", 100, 20, "knn"))

mem_doppel_in_20_20_knn = memUsage(quota_include("all", 20, 20, "knn"))
mem_doppel_in_50_20_knn = memUsage(quota_include("all", 50, 20, "knn"))
mem_doppel_in_70_20_knn = memUsage(quota_include("all", 70, 20, "knn"))
mem_doppel_in_100_20_knn = memUsage(quota_include("all", 100, 20, "knn"))
# =========================================================================================
mem_doppel_same_knn_a = [mem_doppel_same_20_20_knn, mem_doppel_same_50_20_knn, mem_doppel_same_70_20_knn, mem_doppel_same_100_20_knn]
mem_doppel_ex_knn_a = [mem_doppel_ex_20_20_knn, mem_doppel_ex_50_20_knn, mem_doppel_ex_70_20_knn, mem_doppel_ex_100_20_knn]
mem_doppel_in_knn_a = [mem_doppel_in_20_20_knn, mem_doppel_in_50_20_knn, mem_doppel_in_70_20_knn, mem_doppel_in_100_20_knn]
# =========================================================================================

# Memory for doppel KNN b)
mem_doppel_same_100_10_knn = memUsage(quota_same("all", 100, 10, "knn"))
mem_doppel_same_100_15_knn = memUsage(quota_same("all", 100, 15, "knn"))
mem_doppel_same_100_25_knn = memUsage(quota_same("all", 100, 25, "knn"))

mem_doppel_ex_100_10_knn = memUsage(quota_exclude("all", 100, 10, "knn"))
mem_doppel_ex_100_15_knn = memUsage(quota_exclude("all", 100, 15, "knn"))
mem_doppel_ex_100_25_knn = memUsage(quota_exclude("all", 100, 25, "knn"))

mem_doppel_in_100_10_knn = memUsage(quota_include("all", 100, 10, "knn"))
mem_doppel_in_100_15_knn = memUsage(quota_include("all", 100, 15, "knn"))
mem_doppel_in_100_25_knn = memUsage(quota_include("all", 100, 25, "knn"))
# =========================================================================================
mem_doppel_same_knn_b = [mem_doppel_same_100_10_knn, mem_doppel_same_100_15_knn, mem_doppel_same_100_20_knn, mem_doppel_same_100_25_knn]
mem_doppel_ex_knn_b = [mem_doppel_ex_100_10_knn, mem_doppel_ex_100_15_knn, mem_doppel_ex_100_20_knn, mem_doppel_ex_100_25_knn]
mem_doppel_in_knn_b = [mem_doppel_in_100_10_knn, mem_doppel_in_100_15_knn, mem_doppel_in_100_20_knn, mem_doppel_in_100_25_knn]
# =========================================================================================

# Memory for doppel RF a)
mem_doppel_same_20_20_rf = memUsage(quota_same("all", 20, 20, "rf"))
mem_doppel_same_50_20_rf = memUsage(quota_same("all", 50, 20, "rf"))
mem_doppel_same_70_20_rf = memUsage(quota_same("all", 70, 20, "rf"))
mem_doppel_same_100_20_rf = memUsage(quota_same("all", 100, 20, "rf"))

mem_doppel_ex_20_20_rf = memUsage(quota_exclude("all", 20, 20, "rf"))
mem_doppel_ex_50_20_rf = memUsage(quota_exclude("all", 50, 20, "rf"))
mem_doppel_ex_70_20_rf = memUsage(quota_exclude("all", 70, 20, "rf"))
mem_doppel_ex_100_20_rf = memUsage(quota_exclude("all", 100, 20, "rf"))

mem_doppel_in_20_20_rf = memUsage(quota_include("all", 20, 20, "rf"))
mem_doppel_in_50_20_rf = memUsage(quota_include("all", 50, 20, "rf"))
mem_doppel_in_70_20_rf = memUsage(quota_include("all", 70, 20, "rf"))
mem_doppel_in_100_20_rf = memUsage(quota_include("all", 100, 20, "rf"))
# =========================================================================================
mem_doppel_same_rf_a = [mem_doppel_same_20_20_rf, mem_doppel_same_50_20_rf, mem_doppel_same_70_20_rf, mem_doppel_same_100_20_rf]
mem_doppel_ex_rf_a = [mem_doppel_ex_20_20_rf, mem_doppel_ex_50_20_rf, mem_doppel_ex_70_20_rf, mem_doppel_ex_100_20_rf]
mem_doppel_in_rf_a = [mem_doppel_in_20_20_rf, mem_doppel_in_50_20_rf, mem_doppel_in_70_20_rf, mem_doppel_in_100_20_rf]
# =========================================================================================

# Memory for doppel RF b)
mem_doppel_same_100_10_rf = memUsage(quota_same("all", 100, 10, "rf"))
mem_doppel_same_100_15_rf = memUsage(quota_same("all", 100, 15, "rf"))
mem_doppel_same_100_25_rf = memUsage(quota_same("all", 100, 25, "rf"))

mem_doppel_ex_100_10_rf = memUsage(quota_exclude("all", 100, 10, "rf"))
mem_doppel_ex_100_15_rf = memUsage(quota_exclude("all", 100, 15, "rf"))
mem_doppel_ex_100_25_rf = memUsage(quota_exclude("all", 100, 25, "rf"))

mem_doppel_in_100_10_rf = memUsage(quota_include("all", 100, 10, "rf"))
mem_doppel_in_100_15_rf = memUsage(quota_include("all", 100, 15, "rf"))
mem_doppel_in_100_25_rf = memUsage(quota_include("all", 100, 25, "rf"))
# =========================================================================================
mem_doppel_same_rf_b = [mem_doppel_same_100_10_rf, mem_doppel_same_100_15_rf, mem_doppel_same_100_20_rf, mem_doppel_same_100_25_rf]
mem_doppel_ex_rf_b = [mem_doppel_ex_100_10_rf, mem_doppel_ex_100_15_rf, mem_doppel_ex_100_20_rf, mem_doppel_ex_100_25_rf]
mem_doppel_in_rf_b = [mem_doppel_in_100_10_rf, mem_doppel_in_100_15_rf, mem_doppel_in_100_20_rf, mem_doppel_in_100_25_rf]
# =========================================================================================

x_ticks_a = ['20P', '50P', '70P', '100P']
x_ticks_b = ['10C', '15C', '20C', ' 25C']

x_ticks = [x_ticks_a, x_ticks_b]
x_ticks_flat = [item for elem in x_ticks for item in elem]
x_axis_run = np.arange(len(run_features_a) + len(run_features_b) + len(run_doppel_same_svm_a) +
                       len(run_doppel_same_svm_b) + len(run_doppel_ex_svm_a) + len(run_doppel_ex_svm_b) +
                       len(run_doppel_in_svm_a) + len(run_doppel_in_svm_b) + len(run_doppel_same_knn_a) +
                       len(run_doppel_same_knn_b) + len(run_doppel_ex_knn_a) + len(run_doppel_ex_knn_b) +
                       len(run_doppel_in_knn_a) + len(run_doppel_in_knn_b) + len(run_doppel_same_rf_a) +
                       len(run_doppel_same_rf_b) + len(run_doppel_ex_rf_a) + len(run_doppel_ex_rf_b) +
                       len(run_doppel_in_rf_a) + len(run_doppel_in_rf_b))

run = [run_doppel_same_svm_a, run_doppel_same_svm_b, run_doppel_ex_svm_a,
       run_doppel_ex_svm_b, run_doppel_in_svm_a, run_doppel_in_svm_b, run_doppel_same_knn_a, run_doppel_same_knn_b,
       run_doppel_ex_knn_a, run_doppel_ex_knn_b, run_doppel_in_knn_a, run_doppel_in_knn_b, run_doppel_same_rf_a,
       run_doppel_same_rf_b, run_doppel_ex_rf_a, run_doppel_ex_rf_b,
       run_doppel_in_rf_a, run_doppel_in_rf_b]

x_axis_mem = np.arange(len(mem_features_a) + len(mem_features_b) + len(mem_doppel_same_svm_a) +
                       len(mem_doppel_same_svm_b) + len(mem_doppel_ex_svm_a) + len(mem_doppel_ex_svm_b) +
                       len(mem_doppel_in_svm_a) + len(mem_doppel_in_svm_b) + len(mem_doppel_same_knn_a) +
                       len(mem_doppel_same_knn_b) + len(mem_doppel_ex_knn_a) + len(mem_doppel_ex_knn_b) +
                       len(mem_doppel_in_knn_a) + len(mem_doppel_in_knn_b) + len(mem_doppel_same_rf_a) +
                       len(mem_doppel_same_rf_b) + len(mem_doppel_ex_rf_a) + len(mem_doppel_ex_rf_b) +
                       len(mem_doppel_in_rf_a) + len(mem_doppel_in_rf_b))

mem = [mem_doppel_same_svm_a, mem_doppel_same_svm_b, mem_doppel_ex_svm_a,
       mem_doppel_ex_svm_b, mem_doppel_in_svm_a, mem_doppel_in_svm_b, mem_doppel_same_knn_a, mem_doppel_same_knn_b,
       mem_doppel_ex_knn_a, mem_doppel_ex_knn_b, mem_doppel_in_knn_a, mem_doppel_in_knn_b, mem_doppel_same_rf_a,
       mem_doppel_same_rf_b, mem_doppel_ex_rf_a, mem_doppel_ex_rf_b,
       mem_doppel_in_rf_a, mem_doppel_in_rf_b]

colors = ['plum', 'mediumslateblue', 'rebeccapurple', 'mediumorchid', 'violet', 'darkmagenta', 'aquamarine',
          'cadetblue', 'darkslategray', 'teal', 'darkturquoise', 'cadetblue', 'darkorange', 'burlywood', 'moccasin',
          'darkgoldenrod', 'gold', 'khaki']

legends = ['[SVM] Quota Same \n(With 20 Comments)', '[SVM] Quota Same \n(Per 100 Pseudonyms)',
           '[SVM] Quota Exclude \n(With 20 Comments)', '[SVM] Quota Exclude \n(Per 100 Pseudonyms)',
           '[SVM] Quota Include \n(With 20 Comments)', '[SVM] Quota Include \n(Per 100 Pseudonyms)',
           '[KNN] Quota Same \n(With 20 Comments)', '[KNN] Quota Same \n(Per 100 Pseudonyms)',
           '[KNN] Quota Exclude \n(With 20 Comments)', '[KNN] Quota Exclude \n(Per 100 Pseudonyms)',
           '[KNN] Quota Include \n(With 20 Comments)', '[KNN] Quota Include \n(Per 100 Pseudonyms)',
           '[RF] Quota Same \n(With 20 Comments)', '[RF] Quota Same \n(Per 100 Pseudonyms)',
           '[RF] Quota Exclude \n(With 20 Comments)', '[RF] Quota Exclude \n(Per 100 Pseudonyms)',
           '[RF] Quota Include \n(With 20 Comments)', '[RF] Quota Include \n(Per 100 Pseudonyms)']
for r in run:
    print("r: ", r)
for m in mem:
    print("m: ", m)
plt.scatter(x_axis_run[0:4], run_features_a, color='red', label='Feature Extraction \n(With 20 Comments)')
plt.plot(x_axis_run[0:4], run_features_a, color='red')
plt.scatter(x_axis_run[4:8], run_features_b, color='blue', label='Features Extraction \n(Per 100 Pseudonyms)')
plt.plot(x_axis_run[4:8], run_features_b, color='blue')

run_flat = [item for elem in run for item in elem]
for i, j in zip(x_axis_run, range(8, len(x_axis_run), 4)):
    plt.scatter(x_axis_run[j:j+4], run[i], label=legends[i], color=colors[i])
    plt.plot(x_axis_run[j:j + 4], run[i], color=colors[i])

plt.xticks(x_axis_run, x_ticks_flat*10, fontsize=7)

plt.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1, 1))
plt.tick_params(axis='x', rotation=60)
plt.title('Runtime of feature extraction & Doppelgänger Detection')
plt.xlabel('Experiments (P: Pseudonyms / C: Comments)')
plt.ylabel('Runtime (Secs)')
plt.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)
plt.show()


plt.scatter(x_axis_mem[0:4], mem_features_a, color='red', label='Feature Extraction \n(With 20 Comments)')
plt.plot(x_axis_mem[0:4], mem_features_a, color='red')
plt.scatter(x_axis_mem[4:8], mem_features_b, color='blue', label='Features Extraction \n(Per 100 Pseudonyms)')
plt.plot(x_axis_mem[4:8], mem_features_b, color='blue')

mem_flat = [item for elem in mem for item in elem]
for i, j in zip(x_axis_mem, range(8, len(x_axis_mem), 4)):
    plt.scatter(x_axis_mem[j:j+4], mem[i], label=legends[i], color=colors[i])
    plt.plot(x_axis_mem[j:j + 4], mem[i], color=colors[i])

plt.xticks(x_axis_mem, x_ticks_flat*10, fontsize=7)

plt.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1, 1))
plt.tick_params(axis='x', rotation=60)
plt.title('Memory Usage of feature extraction & Doppelgänger Detection')
plt.xlabel('Experiments (P: Pseudonyms / C: Comments)')
plt.ylabel('Memory Usage (MB)')
plt.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1)
plt.show()
