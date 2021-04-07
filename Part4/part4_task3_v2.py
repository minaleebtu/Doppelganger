import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from part4_task2_v2 import numOfP, numOfCommPerP

pd.options.display.max_columns = None

# File containing values of Username, Comment, 11 Features, Label of user
new = pd.read_csv('Joined_v2.csv')
new = new.drop(new.columns[new.columns.str.contains('unnamed', case=False)], axis=1)


# Get usernames, labels of users, feature values of exact number of users and comments
def selectData(numOfP, numOfComm):
    numOfUser = int(numOfP / 2)
    numOfComm = int(numOfComm * 2)
    result = pd.DataFrame()

    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn].drop(['articleTitle', 'comment'], axis=1)[0:numOfComm],
                               ignore_index=True)

    result = result.replace(np.nan, '', regex=True)
    # result = result.drop(['username', 'Label'], axis=1)

    return result


def euclidDoppel(selectedData):
    authors_name = list(set(selectedData['username'].values.tolist()))
    # print("authors_name: ", authors_name)
    userLabel = selectedData['Label'].to_numpy()
    features = selectedData.drop(['username', 'Label'], axis=1).to_numpy()

    encode_to_num = pd.Series(userLabel, userLabel).to_dict()
    allAuthors = list(encode_to_num.keys())

    cols = ['Author 1', 'Author 2', 'Encode 1', 'Encode 2', 'Euclidean Distance']

    resultList = []

    for i in range(len(allAuthors)):
        a = int(allAuthors[i])
        for j in range(i + 1, len(allAuthors)):
            b = allAuthors[j]

            resultList.append([authors_name[a], authors_name[b], encode_to_num[a], encode_to_num[b],
                               euclidean(features[a], features[b])])

    result = pd.DataFrame(resultList, columns=cols)

    # Get threshold as an input
    while True:
        threshold = input("Please enter the threshold (range:" + str("{:.2f}".format(result['Euclidean Distance'].min()))
                          + "-" + str("{:.2f}".format(result['Euclidean Distance'].max())) + "): ")

        # Check the input parameter is number or not. If parameter is not number, make user input threshold again
        try:
            threshold = int(threshold)
            break
        except ValueError:
            try:
                threshold = float(threshold)
                break
            except ValueError:
                print("This is not a number. Please enter a valid number")

    result['Euclidean Doppelgangers'] = result['Euclidean Distance'].apply(lambda d: 1 if d < threshold else 0)

    return result

# def supervised():


# Function for calculating the Euclidean score
def predict_score(data):
#     data = pd.read_csv(df)
#     euclidDoppel(selectData)
    X = data.drop(['Euclidean Doppelgangers', 'Author 1', 'Author 2'], axis=1)
    y = data['Euclidean Doppelgangers']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = svm.SVC()
#     clf = RandomForestClassifier(n_estimators=40)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    return score


print("euclidDoppel:\n", euclidDoppel(selectData(20, 20)))

# def plot_all():
dopp_a = numOfP
dopp_b = numOfCommPerP
euclid_a = []
euclid_b = []

euclid_a.append(predict_score(euclidDoppel(selectData(20, 20))))
euclid_a.append(predict_score(euclidDoppel(selectData(40, 20))))
euclid_a.append(predict_score(euclidDoppel(selectData(60, 20))))

euclid_b.append(predict_score(euclidDoppel(selectData(60, 10))))
euclid_b.append(predict_score(euclidDoppel(selectData(60, 20))))
euclid_b.append(predict_score(euclidDoppel(selectData(60, 30))))

print("dopp_a", dopp_a)
print("dopp_b", dopp_b)
print("euclid_a", euclid_a)
print("euclid_b", euclid_b)


print('-'*80)
print('Doppelganger Accuracy Score With 20 Pseudonyms & 20 Comments : ', dopp_a[0])
print('Euclidean Accuracy Score With 20 Pseudonyms & 20 Comments :', euclid_a[0])
print('-'*80)

print('Doppelganger Accuracy Score With 40 Pseudonyms & 20 Comments :', dopp_a[1])
print('Euclidean Accuracy Score With 40 Pseudonyms & 20 Comments :', euclid_a[1])
print('-'*80)

print('Doppelganger Accuracy Score With 60 Pseudonyms & 20 Comments :', dopp_a[2])
print('Euclidean Accuracy Score With 60 Pseudonyms & 20 Comments :', euclid_a[2])
print('-'*80)

print('Doppelganger Accuracy Score With 60 Pseudonyms & 10 Comments :', dopp_b[0])
print('Euclidean Accuracy Score With 60 Pseudonyms & 10 Comments :', euclid_b[0])
print('-'*80)

print('Doppelganger Accuracy Score With 60 Pseudonyms & 20 Comments :', dopp_b[1])
print('Euclidean Accuracy Score With 60 Pseudonyms & 20 Comments :', euclid_b[1])
print('-'*80)

print('Doppelganger Accuracy Score With 60 Users & 30 Comments :', dopp_b[2])
print('Euclidean Accuracy Score With 60 Users & 30 Comments :', euclid_b[2])

x_ticks = ['20 Users', '40 Users', '60 Users', '10 Comments', '20 Comments', '30 Comments']
x_axis = np.arange(6)

barWidth = 0.3
r1 = np.arange(len(x_axis))
r2 = [x + barWidth for x in r1]

plt.bar(r1[0:3], dopp_a, width=barWidth, color='lightblue', capsize=7, label='Doppelgaenger/With 20 Comments')
plt.bar(r2[0:3], euclid_a, width=barWidth, color='royalblue', capsize=7, label='Euclidean/With 20 Comments')

plt.bar(r1[3:6], dopp_b, width=barWidth, color='salmon', capsize=7, label='Doppelgaenger/Per 60 Users')
plt.bar(r2[3:6], euclid_b, width=barWidth, color='maroon', capsize=7, label='Euclidean/Per 60 Users')

plt.xticks([r + barWidth for r in range(6)], x_ticks, fontsize=6)

plt.legend(loc='lower center')
plt.xlabel('Experiments')
plt.ylabel('Accuracy score')
plt.show()
