import random
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
from sklearn.metrics import f1_score, accuracy_score

# File containing values of Username, Comment, 11 Features, Label of user
new = pd.read_csv('Joined_v2.csv')
new = new.drop(new.columns[new.columns.str.contains('unnamed', case=False)], axis=1)

kfold = GroupKFold(n_splits=3)

# pd.options.display.max_columns = None
while True:
    selection = input(
        "Please enter the classifiers('SVM' for svm , 'KNN' for KN Neighbors, 'RF' for Random Forest): ")

    if selection.upper() not in ('SVM', 'KNN', 'RF'):
        print("Not an appropriate choice. Enter valid value ('SVM', 'KNN', 'RF')")
    else:
        break

if selection.upper() == 'SVM':
    clf = svm.SVC(kernel='linear', probability=True)
elif selection.upper() == 'KNN':
    clf = KNeighborsClassifier(n_neighbors=3)
elif selection.upper() == 'RF':
    clf = RandomForestClassifier(n_estimators=40)

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


def getCombinedProbs(y_arr, prob_per_author):
    avg_prob = []

    y_arr_list = list(set(y_arr.tolist()))

    for i in range(len(y_arr_list)):
        for j in range(i + 1, len(y_arr_list)):
            avg = (prob_per_author[i][j] + prob_per_author[j][i]) / 2

            avg_prob.append(avg)

    return np.array(avg_prob)


# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')


def quota_same(scenario, numOfP, numOfComm):
    split_none = selectData(numOfP, numOfComm)
    split_all = splitAll(selectData(numOfP, numOfComm))
    if scenario == 'none':
        X = getPca(split_none.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_none)
    elif scenario == 'all':
        X = getPca(split_all.drop(['articleTitle', 'username', 'comment', 'Label'], axis=1))
        y = getLabel(split_all)

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


def quota_exclude(scenario, numOfP, numOfComm):
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


def quota_include(scenario, numOfP, numOfComm):
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


print('='*80)
print('Scenario "None" (' + selection.upper() + ')')
scenario_none_a = [quota_same("none", 50, 20), quota_same("none", 100, 20)]
print("* When the quota of doppelgängers in the training and the testing sets is the same *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_none_a[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_none_a[1])
print('-'*80)
scenario_none_b = [quota_exclude("none", 50, 20), quota_exclude("none", 100, 20)]
print("* When excluding the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_none_b[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_none_b[1])
print('-'*80)
scenario_none_c = [quota_include("none", 50, 20), quota_include("none", 100, 20)]
print("* When including the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_none_c[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_none_c[1])
print('-'*80)
print('='*80)

print('Scenario "Single" (' + selection.upper() + ')')
scenario_single_b = [quota_exclude("single", 50, 20), quota_exclude("single", 100, 20)]
print("* When excluding the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_single_b[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_single_b[1])
print('-'*80)
scenario_single_c = [quota_include("single", 50, 20), quota_include("single", 100, 20)]
print("* When including the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_single_c[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_single_c[1])
print('-'*80)
print('='*80)

print('Scenario "Random" (' + selection.upper() + ')')
scenario_random_b = [quota_exclude("random", 50, 20), quota_exclude("random", 100, 20)]
print("* When excluding the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_random_b[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_random_b[1])
print('-'*80)
scenario_random_c = [quota_include("random", 50, 20), quota_include("random", 100, 20)]
print("* When including the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_random_c[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_random_c[1])
print('-'*80)
print('='*80)

print('Scenario "All" (' + selection.upper() + ')')
scenario_all_a = [quota_same("all", 50, 20), quota_same("all", 100, 20)]
print("* When the quota of doppelgängers in the training and the testing sets is the same *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_all_a[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_all_a[1])
print('-'*80)
scenario_all_b = [quota_exclude("all", 50, 20), quota_exclude("all", 100, 20)]
print("* When excluding the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_all_b[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_all_b[1])
print('-'*80)
scenario_all_c = [quota_include("all", 50, 20), quota_include("all", 100, 20)]
print("* When including the quota of doppelgängers in the testing fold *")
print("Accuracy Score With 50 Pseudonyms & 20 Comments : ", scenario_all_c[0])
print("Accuracy Score With 100 Pseudonyms & 20 Comments : ", scenario_all_c[1])
print('-'*80)
print('='*80)

x_ticks = ['50 P & 20 Comm', '100 P & 20 Comm']

x_axis_none = np.arange(len(scenario_none_a) + len(scenario_none_b) + len(scenario_none_c))
x_axis_single = np.arange(len(scenario_single_b) + len(scenario_single_c))
x_axis_random = np.arange(len(scenario_random_b) + len(scenario_random_c))
x_axis_all = np.arange(len(scenario_all_a) + len(scenario_all_b) + len(scenario_all_c))

plt.bar(x_axis_none[0:2], scenario_none_a, color='rosybrown', label='Quota Same')
plt.bar(x_axis_none[2:4], scenario_none_b, color='indianred', label='Quota Exclude')
plt.bar(x_axis_none[4:6], scenario_none_c, color='maroon', label='Quota Include')
plt.xticks(x_axis_none, [x for x in x_ticks] * 3, fontsize=6)

scenario_none = [scenario_none_a, scenario_none_b, scenario_none_c]
scenario_none_flat = [item for elem in scenario_none for item in elem]
for i, v in enumerate(scenario_none_flat):
    plt.text(i, v, '%.3f' % v, ha='center', va='bottom')

plt.legend(loc='lower center')
plt.title('Scenario "None" (' + selection.upper() + ')')
plt.xlabel('Experiments')
plt.ylabel('Accuracy score')
plt.show()

plt.bar(x_axis_single[0:2], scenario_single_b, color='rosybrown', label='Quota Exclude')
plt.bar(x_axis_single[2:4], scenario_single_c, color='indianred', label='Quota Include')
plt.xticks(x_axis_single, [x for x in x_ticks] * 2, fontsize=6)

scenario_single = [scenario_single_b, scenario_single_c]
scenario_single_flat = [item for elem in scenario_single for item in elem]
for i, v in enumerate(scenario_single_flat):
    plt.text(i, v, '%.3f' % v, ha='center', va='bottom')

plt.legend(loc='lower center')
plt.title('Scenario "Single" (' + selection.upper() + ')')
plt.xlabel('Experiments')
plt.ylabel('Accuracy score')
plt.show()

plt.bar(x_axis_random[0:2], scenario_random_b, color='rosybrown', label='Quota Exclude')
plt.bar(x_axis_random[2:4], scenario_random_c, color='indianred', label='Quota Include')
plt.xticks(x_axis_random, [x for x in x_ticks] * 2, fontsize=6)

scenario_random = [scenario_random_b, scenario_random_c]
scenario_random_flat = [item for elem in scenario_random for item in elem]
for i, v in enumerate(scenario_random_flat):
    plt.text(i, v, '%.3f' % v, ha='center', va='bottom')

plt.legend(loc='lower center')
plt.title('Scenario "Random" (' + selection.upper() + ')')
plt.xlabel('Experiments')
plt.ylabel('Accuracy score')
plt.show()

plt.bar(x_axis_all[0:2], scenario_all_a, color='rosybrown', label='Quota Same')
plt.bar(x_axis_all[2:4], scenario_all_b, color='indianred', label='Quota Exclude')
plt.bar(x_axis_all[4:6], scenario_all_c, color='maroon', label='Quota Include')
plt.xticks(x_axis_all, [x for x in x_ticks] * 3, fontsize=6)

scenario_all = [scenario_all_a, scenario_all_b, scenario_all_c]
scenario_all_flat = [item for elem in scenario_all for item in elem]
for i, v in enumerate(scenario_all_flat):
    plt.text(i, v, '%.3f' % v, ha='center', va='bottom')

plt.legend(loc='lower center')
plt.title('Scenario "All" (' + selection.upper() + ')')
plt.xlabel('Experiments')
plt.ylabel('Accuracy score')
plt.show()
