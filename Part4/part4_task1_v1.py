# optimal threshold for precision-recall curve with logistic regression model
import itertools

from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score
from matplotlib import pyplot as plt
from sklearn import svm
import pandas as pd
import numpy as np


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print("\n", cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


new = pd.read_csv('resultss.csv')

rtr = new.drop(['Encode 1', 'Encode 2'], axis=1)
# Getting mean of combined probabilities (multipliction, average, squared average) per row as threshold
rtr['Threshold'] = rtr.apply(lambda row: (row.Multiplication + row.Averaged + row.Squared) / 3, axis=1)

# Getting mean of threshold from all rows and compare with each threshold
# if threshold is greater than mean of threshold from all rows, we can say two authors are doppelgangers (label: 1)
rtr['Doppelgangers'] = rtr['Threshold'].apply(lambda x: 0 if x < rtr['Threshold'].mean() else 1)

# Values of probabilities (dependent data)
X = rtr.drop(['Author 1', ' Author 2', 'Doppelgangers'], axis=1).to_numpy()

# Label of doppelganger (independent data)
y = rtr['Doppelgangers'].to_numpy()

# Splitting data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = svm.SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

def to_labels(pos_probs, threshold):
    # print("result: ", (pos_probs >= threshold).astype('int'), type((pos_probs >= threshold).astype('int')), len((pos_probs >= threshold).astype('int')))
    return (pos_probs >= threshold).astype('int')

    # evaluate each threshold
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
# fit a model
# model = LogisticRegression(solver='lbfgs')
model = svm.SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# predict probabilities
yhat = model.predict_proba(testX)
# keep probabilities for the positive outcome only
probs = yhat[:, 1]
# print("yhat: ", yhat, yhat.shape)
# print("probs: ", probs, probs.shape)
# define thresholds
thresholds = np.arange(0, 1, 0.001)
# print("thresholds: ", thresholds, thresholds.shape)
# # evaluate each threshold
scores = [f1_score(testy, to_labels(probs, t), average='micro') for t in thresholds]
# # get best threshold
index = np.argmax(scores)
print('Threshold=%.3f, F-Score=%.5f' % (thresholds[index], scores[index]))
# precision, recall, thresholds = precision_recall_curve(testy.ravel(), probs.ravel())
#
# fscore = (2 * precision * recall) / (precision + recall)
# index = np.argmax(fscore)
# print('Best Threshold: ', thresholds[index], '& F-Score: ', fscore[index])

# predict probabilities
yhat = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
yhat = yhat[:, 1]
# calculate roc curves
precision, recall, thresholds = precision_recall_curve(y_test, yhat)
# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
# scores = [f1(y_test, to_labels(yhat, t), average='micro') for t in thresholds]

# locate the index of the largest f score
ix = argmax(fscore)
print('Best Threshold=%.2f, F-Score=%.2f' % (thresholds[ix], fscore[ix]))

# extracting true_positives, false_positives, true_negatives, false_negatives
print("y_test: ", y_test, y_test.shape)
print("yhat.round(): ", yhat.round().shape)
tn, fp, fn, tp = confusion_matrix(y_test, yhat.round()).ravel()
print("True Negatives: ", tn)
print("False Positives: ", fp)
print("False Negatives: ", fn)
print("True Positives: ", tp)
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat.round())
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Negative', 'Positive'],
                      title='Confusion matrix')

# Accuracy
Accuracy = (tn + tp) * 100 / (tp + tn + fp + fn)
print("Accuracy {:0.2f}".format(Accuracy))

# Precision
Precision = tp / (tp + fp)
print("Precision {:0.2f}".format(Precision))

# Recall
Recall = tp / (tp + fn)
print("Recall {:0.2f}".format(Recall))

# F1 Score
f1 = (2 * Precision * Recall) / (Precision + Recall)
print("F1 Score {:0.2f}".format(f1))
