import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


new = pd.read_csv('resultss.csv')
pd.options.display.width = 0

pd.options.display.float_format = "{:,.3f}".format

rtr = new.drop([ 'Encode 1', 'Encode 2'], axis = 1)
rtr['Threshold'] = rtr.mean(axis=1)
np.where(rtr["Threshold"] == rtr["col2"], True, False)
# result.drop([''])
# rtr['Sum_Of_Prob'] = rtr.apply(lambda row: row.Multiplication+ row.Averaged+row.Squared, axis=1)
# rtr['Doppelgangers'] = rtr['Threshold'].apply(lambda x: '0' if x <= rtr['Threshold'] else '1')

X = rtr.drop(['Author 1', ' Author 2','Doppelgangers'], axis = 1)
y = rtr['Doppelgangers']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)



classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)
print(cm.shape)

# clf.fit(X_train, y_train)
# SVC(random_state=0)
import matplotlib.pyplot as plt
plot_confusion_matrix(classifier, X_test, y_test)
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)

# Generating accuracy, precision, recall and f1-score
from sklearn.metrics import classification_report
target_names = ['Doppelganger', 'Non-Doppelganger']
print(classification_report(y_test, y_pred, target_names=target_names))

# print(new)
# print(os.getcwd())


