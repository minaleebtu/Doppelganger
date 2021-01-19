import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
# Generating accuracy, precision, recall and f1-score
from sklearn.metrics import classification_report

new = pd.read_csv('resultss.csv')

rtr = new.drop(['Encode 1', 'Encode 2'], axis=1)
# Getting mean of combined probabilities (multipliction, average, squared average) per row as threshold
rtr['Threshold'] = rtr.apply(lambda row: (row.Multiplication + row.Averaged + row.Squared)/3, axis=1)
# Getting mean of threshold from all rows and compare with each threshold
# if threshold is greater than mean of threshold from all rows, we can say two authors are doppelgangers (label: 1)
rtr['Doppelgangers'] = rtr['Threshold'].apply(lambda x: 0 if x < rtr['Threshold'].mean() else 1)

# Values of probabilities (dependent data)
X = rtr.drop(['Author 1', ' Author 2', 'Doppelgangers'], axis=1)
# Label of doppelganger (independent data)
y = rtr['Doppelgangers']

# Splitting data into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Defining classifier and train dataset
classifier = svm.SVC()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Getting the number of true negative, false positive, false negative, true positive
cm = confusion_matrix(y_test, y_pred)
print("True Negative: ", cm[0][0], "\nFalse Positive: ", cm[0][1])
print("False Negative: ", cm[1][0], "\nTrue Positive: ", cm[1][1])
print("====================================================================================")

# Plotting the result
plot_confusion_matrix(classifier, X_test, y_test)
plt.show()

# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print(tn, fp, fn, tp)

# Values of precision, recall, F1 score for each class
target_names = ['Non-Doppelganger', 'Doppelganger']
print(classification_report(y_test, y_pred, target_names=target_names))
