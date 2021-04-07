from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import LeaveOneGroupOut
import joblib
from joblib import Parallel, delayed
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# File containing values of Username, Comment, 11 Features, Label of user
new = pd.read_csv('Joinedv1.csv')
datdrop = new.drop(['Unnamed: 0', 'username', 'comment', 'Label'], axis = 1)

# Converting the dataset into an array
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
dat_array = datdrop.to_numpy()


# Calculating the PCA to reduce number of features
def getPca(dat_array):
    pca = PCA(n_components=0.999)
    DATA_PCA = pca.fit_transform(dat_array)

    return DATA_PCA


# Getting label value of exact number of users as exact number of comments
def getLabel(numOfUser, numOfComm):
    result = pd.DataFrame()

    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn][0:numOfComm], ignore_index=True)

    newResult = result['Label'].to_numpy()

    return newResult


# Getting username of exact number of users
def getUserName(numOfUser):
    result = pd.DataFrame()

    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn], ignore_index=True)

    # Removing duplicated usernames
    newResult = list(set(result['username'].to_list()))

    return newResult


# Getting feature values of exact number of users and comments
def selectData(numOfUser, numOfComm):
    result = pd.DataFrame()

    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn].drop(['Unnamed: 0', 'comment'], axis=1)[0:numOfComm],
                               ignore_index=True)

    result = result.drop(['username', 'Label'], axis=1)

    return result.to_numpy()


# Separating the dependant and independent variable per experiment
X_20_20 = getPca(selectData(20, 20))
# X_40_20 = getPca(selectData(40, 20))
# X_60_20 = getPca(selectData(60, 20))
# X_60_10 = getPca(selectData(60, 10))
# X_60_30 = getPca(selectData(60, 30))

y_20_20 = getLabel(20, 20)
# y_40_20 = getLabel(40, 20)
# y_60_20 = getLabel(60, 20)
# y_60_10 = getLabel(60, 10)
# y_60_30 = getLabel(60, 30)

# Label of users (dictionary type)
encode_to_num_20 = pd.Series(getLabel(20, 1), getLabel(20, 1)).to_dict()
# encode_to_num_40 = pd.Series(getLabel(40, 1), getLabel(40, 1)).to_dict()
# encode_to_num_60 = pd.Series(getLabel(60, 1), getLabel(60, 1)).to_dict()

# Label of users
allAuthors_20 = encode_to_num_20.keys()
# allAuthors_40 = encode_to_num_40.keys()
# allAuthors_60 = encode_to_num_60.keys()

# Define the classifiers
clf = svm.SVC(kernel='linear', probability=True)


# Define a function of separating train and test data, creating models per user
def getProbsThread(nthread, clf, data, label, allAuthors, modeldir, saveModel):
    crossval = LeaveOneGroupOut()

    crossval.get_n_splits(groups=label)

    prob_per_author = [[0] * (len(allAuthors)) for i in range(len(allAuthors))]

    scores = Parallel(n_jobs=nthread)(
        delayed(getProbsTrainTest)(clf, data, label, train, test, modeldir, saveModel) for train, test in
        crossval.split(data, label, groups=label))

    for train, test in crossval.split(data, label, groups=label):

        anAuthor = int(label[test[0]])
        train_data_label = label[train]
        trainAuthors = list(set(train_data_label))
        # test_data_label = label[test]
        nTestDoc = len(scores)  # len(test_data_label)
        for j in range(nTestDoc):
            for i in range(len(trainAuthors)):
                try:
                    prob_per_author[anAuthor][int(trainAuthors[i])] += scores[anAuthor - 1][j][i]
                except IndexError:
                    continue

        for i in range(len(trainAuthors)):
            prob_per_author[anAuthor][int(trainAuthors[i])] /= nTestDoc
    return prob_per_author


# Create a function for calculating the probability of training and test data
def getProbsTrainTest(clf, data, label, train, test, modeldir, saveModel):
    anAuthor = int(label[test[0]])

    train_data = data[train, :]
    train_data_label = label[train]

    # test on anAuthor
    test_data = data[test, :]

    # check if we already have a model
    modelFile = modeldir + str(anAuthor) + "-" + saveModel

    if os.path.exists(modelFile):
        clf = joblib.load(modelFile)
    else:
        # train
        clf.fit(train_data, train_data_label)

        # save model
        joblib.dump(clf, modelFile, compress=9)
        print("model saved: ", modelFile)

    # get probabilities
    scores = clf.predict_proba(test_data)
    return scores


def getCombinedProbs(outfile, prob_per_author, allAuthors, encode_to_num, authors_name):
    total_prob = {}
    add_prob = {}
    sq_prob = {}
    print("prob_per_author: ", prob_per_author, len(prob_per_author))
    print("allAuthors: ", allAuthors, type(allAuthors))

    with open(outfile, "w+", encoding='utf-8') as out:
        out.write(
            'Author 1,Author 2,P(A->B),P(B->A),Multiplication,Averaged,Squared,Encode 1,Encode 2\n')
        for i in range(len(allAuthors)):
            a = int(allAuthors[i])
            for j in range(i + 1, len(allAuthors)):
                b = allAuthors[j]

                result = 0

                total = prob_per_author[a][b] * prob_per_author[b][a]
                addition = (prob_per_author[a][b] + prob_per_author[b][a]) / 2
                sqsum = (prob_per_author[a][b] * prob_per_author[a][b] + prob_per_author[b][a] * prob_per_author[b][
                    a]) / 2

                out.write(str(authors_name[a]) + "," + str(authors_name[b]) + "," +
                          str(prob_per_author[a][b]) + "," + str(prob_per_author[b][a]) + "," +
                          str(total) + "," + str(addition) + "," + str(sqsum) + "," +
                          str(encode_to_num[a]) + "," + str(encode_to_num[b]) +
                          "\n")

                if total in total_prob.keys():
                    total_prob[total] += result
                else:
                    total_prob[total] = result

                if addition in add_prob.keys():
                    add_prob[addition] += result
                else:
                    add_prob[addition] = result
                if sqsum in sq_prob.keys():
                    sq_prob[sqsum] += result
                else:
                    sq_prob[sqsum] = result

    out.close()

    return total_prob, add_prob, sq_prob


# Getting value of doppelganger detection and append it to csv file
def doppelganger(outfile):
    data = pd.read_csv(outfile)
    pd.options.display.width = 0

    # Getting mean of combined probabilities (multiplication, average, squared average) per row as threshold
    data['Threshold'] = data.apply(lambda row: (row.Multiplication + row.Averaged + row.Squared) / 3, axis=1)
    # Getting mean of threshold from all rows and compare with each threshold
    # if threshold is greater than mean of threshold from all rows, we can say two authors are doppelgangers (label: 1)
    data['Doppelgangers'] = data['Threshold'].apply(lambda th: 0 if th < data['Threshold'].mean() else 1)

    return data.to_csv(outfile, index=False)


# Getting cross validation score
def crossval(df):
    data = pd.read_csv(df)

    X = data.drop(['Doppelgangers', 'Author 1', 'Author 2'], axis=1)
    y = data['Doppelgangers']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    clf = svm.SVC()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    predict = cross_val_score(clf, X, y, cv=3)

    return predict.tolist()


# Creating the probability of each author
prob_per_author_20_20 = getProbsThread(3, clf, X_20_20, y_20_20, allAuthors_20, 'models/20_20/', '100-w10-classifier.joblib.pkl')
# prob_per_author_40_20 = getProbsThread(3, clf, X_40_20, y_40_20, allAuthors_40, 'models/40_20/', '100-w10-classifier.joblib.pkl')
# prob_per_author_60_20 = getProbsThread(3, clf, X_60_20, y_60_20, allAuthors_60, 'models/60_20/', '100-w10-classifier.joblib.pkl')
# prob_per_author_60_10 = getProbsThread(3, clf, X_60_10, y_60_10, allAuthors_60, 'models/60_10/', '100-w10-classifier.joblib.pkl')
# prob_per_author_60_30 = getProbsThread(3, clf, X_60_30, y_60_30, allAuthors_60, 'models/60_30/', '100-w10-classifier.joblib.pkl')

# Result of getting probability values
getCombinedProbs("result_20_20.csv", prob_per_author_20_20, list(allAuthors_20), encode_to_num_20, getUserName(20))
# getCombinedProbs("result_40_20.csv", prob_per_author_40_20, list(allAuthors_40), encode_to_num_40, getUserName(40))
# getCombinedProbs("result_60_20.csv", prob_per_author_60_20, list(allAuthors_60), encode_to_num_60, getUserName(60))
# getCombinedProbs("result_60_10.csv", prob_per_author_60_10, list(allAuthors_60), encode_to_num_60, getUserName(60))
# getCombinedProbs("result_60_30.csv", prob_per_author_60_30, list(allAuthors_60), encode_to_num_60, getUserName(60))

# Appending doppelganger detection values
doppelganger("result_20_20.csv")
# doppelganger("result_40_20.csv")
# doppelganger("result_60_20.csv")
# doppelganger("result_60_10.csv")
# doppelganger("result_60_30.csv")

print("a) Number of pseudonyms (20 comments per pseudonym)")
print("\t20 pseudonyms: ", crossval("result_20_20.csv"))
# print("\t40 pseudonyms: ", crossval("result_40_20.csv"))
# print("\t60 pseudonyms: ", crossval("result_60_20.csv"))
# print("====================================================================================")
# print("b) Number of comments per pseudonym (60 pseudonyms)")
# print("\t10 comments per pseudonym: ", crossval("result_60_10.csv"))
# print("\t20 comments per pseudonym: ", crossval("result_60_20.csv"))
# print("\t30 comments per pseudonym: ", crossval("result_60_30.csv"))

# Plotting the result of point a
# x = [1, 2, 3]
#
# plt.plot(x, crossval("result_20_20.csv"), label='1st stage')
# plt.plot(x, crossval("result_40_20.csv"), label='2nd stage')
# plt.plot(x, crossval("result_60_20.csv"), label='3rd stage')
#
# plt.xlabel('Number of Experiments')
# plt.ylabel('Value of Cross Validation')
#
# plt.title("Cross Validation A")
# # Placing a legend on the axes
# plt.legend()
#
# plt.show()
#
#
# # Plotting the result of point b
# plt.plot(x, crossval("result_60_10.csv"), label='1st stage')
# plt.plot(x, crossval("result_60_20.csv"), label='2nd stage')
# plt.plot(x, crossval("result_60_30.csv"), label='3rd stage')
#
# plt.xlabel('Number of Experiments')
# plt.ylabel('Value of Cross Validation')
#
# plt.title("Cross Validation B")
# plt.legend()
#
# plt.show()
