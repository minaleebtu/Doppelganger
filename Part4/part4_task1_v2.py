from numpy import argmax
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.decomposition import PCA

new = pd.read_csv('Joined_v2.csv')
new = new.drop(new.columns[new.columns.str.contains('unnamed', case=False)], axis=1)

# Calculating the PCA to reduce number of features
def getPca(dat_array):
    pca = PCA(n_components=0.999)
    DATA_PCA = pca.fit_transform(dat_array)

    return DATA_PCA


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


X = new.drop(['username', 'articleTitle', 'comment', 'Label'], axis=1)
y = new['Label']

# Splitting data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# predict probabilities
yhat = model.predict_proba(X_test)
combinedProbs = getCombinedProbs(y_test, yhat.tolist())

trueY = trueY_label(y_test)
thresholds = np.arange(0, 1, 0.001)
fscore = [f1_score(trueY, to_labels(combinedProbs, t), average='micro') for t in thresholds]

# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%.2f, when F-Score=%.2f' % (thresholds[ix], fscore[ix]))
print('=' * 80)
y_pred = (getCombinedProbs(y_test, yhat.tolist()) >= thresholds[ix]).astype('int')

# extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(trueY, y_pred.round(),  labels=[0,1]).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)
print('=' * 80)
# np.set_printoptions(precision=2)

# Accuracy
Accuracy = (tn + tp) * 100 / (tp + tn + fp + fn)
print("Accuracy {:0.2f}".format(Accuracy))

# Precision
Precision = np.nan_to_num(tp / (tp + fp))
print("Precision {:0.2f}".format(Precision))

# Recall
Recall = np.nan_to_num(tp / (tp + fn))
print("Recall {:0.2f}".format(Recall))

# F1 Score
f1 = np.nan_to_num((2 * Precision * Recall) / (Precision + Recall))
print("F1 Score {:0.2f}".format(f1))
