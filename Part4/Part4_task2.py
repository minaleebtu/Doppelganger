from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import LeaveOneGroupOut
import joblib
from joblib import Parallel, delayed
from sklearn import svm

new = pd.read_csv('Joinedv1.csv')
datdrop = new.drop(['Unnamed: 0', 'username', 'comment', 'Label'], axis = 1)

# Convert the dataset into an array
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
dat_array = datdrop.to_numpy()

def getPca(dat_array):
    # Calculate the PCA and extract the features
    pd.options.display.width = 0
    pca = PCA(n_components = 0.999)
    DATA_PCA = pca.fit_transform(dat_array)
    # print("PCA Data : \n", DATA_PCA, DATA_PCA.shape)
    return DATA_PCA


def getLabel(numOfUser, numOfComm):
    result = pd.DataFrame()
    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn][0:numOfComm], ignore_index=True)
    newResult = result['Label'].to_numpy()
    return newResult


def getUserName(numOfUser):
    result = pd.DataFrame()
    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn], ignore_index = True)
    newResult = list(set(result['username'].to_list()))
    return newResult


def selectData(numOfUser, numOfComm):
    result = pd.DataFrame()
    for userIn in range(0, numOfUser):
        result = result.append(new[new.Label == userIn].drop(['Unnamed: 0', 'comment'], axis=1)[0:numOfComm], ignore_index = True)
    result = result.drop(['username', 'Label'], axis=1)

    return result.to_numpy()

# print("getPca:\n", getPca(selectData(40, 30)), getPca(selectData(40, 30)).shape)

# Separating the dependant and independent variable
X_20_20 = getPca(selectData(20, 20))
X_40_20 = getPca(selectData(40, 20))
X_60_20 = getPca(selectData(60, 20))
X_60_10 = getPca(selectData(60, 10))
X_60_30 = getPca(selectData(60, 30))

y_20_20 = getLabel(20, 20)
y_40_20 = getLabel(40, 20)
y_60_20 = getLabel(60, 20)
y_60_10 = getLabel(60, 10)
y_60_30 = getLabel(60, 30)


# Create the probability of each author
# prob_per_author_20_20 = [[0]*(len(y_20_20)) for i in range(len(y_20_20))]

# Convert the dataframe into a dictionary to get the values and keys
# authors_to_num = pd.Series(new["Label"].values,index=new.username).to_dict()
# print(authors_to_num)

encode_to_num_20_20 = pd.Series(getLabel(20,20), getLabel(20,20)).to_dict()
encode_to_num_40_20 = pd.Series(getLabel(40,20), getLabel(40,20)).to_dict()
encode_to_num_60_20 = pd.Series(getLabel(60,20), getLabel(60,20)).to_dict()
encode_to_num_60_10 = pd.Series(getLabel(60,10), getLabel(60,10)).to_dict()
encode_to_num_60_30 = pd.Series(getLabel(60,30), getLabel(60,30)).to_dict()

# print('Total authors: ', len(encode_to_num_20_20.keys()))
# print('Authors are: ', encode_to_num_20_20.values())

allAuthors_20_20 = encode_to_num_20_20.keys()
allAuthors_40_20 = encode_to_num_40_20.keys()
allAuthors_60_20 = encode_to_num_60_20.keys()
allAuthors_60_10 = encode_to_num_60_10.keys()
allAuthors_60_30 = encode_to_num_60_30.keys()


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

                out.write(str(authors_name[a]) + " ," + str(authors_name[b]) + " ," +
                          str(prob_per_author[a][b]) + "," + str(prob_per_author[b][a]) + "," +
                          str(total) + "," + str(addition) + "," + str(sqsum) + "," +
                          str(encode_to_num[a]) + " ," + str(encode_to_num[b]) +
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


def doppelganger(outfile):
    data = pd.read_csv(outfile)
    pd.options.display.width = 0
    pd.options.display.float_format = "{:,.3f}".format

    data = data.drop(['Encode 1', 'Encode 2'], axis=1)
    data['Threshold'] = data.apply(lambda row: (row.Multiplication + row.Averaged + row.Squared) / 3, axis=1)

    data['Doppelgangers'] = data['Threshold'].apply(lambda x: 0 if x <= round(data['Threshold'].mean(), 3) else 1)

    return data.to_csv(outfile, index=False)

# allAuthorNames = authors_to_num.keys()
# print('Valid Author List : ',*list(allAuthorNames), sep = "\n")
prob_per_author_20_20 = getProbsThread(3, clf, X_20_20, y_20_20, allAuthors_20_20, 'models/20_20/', '100-w10-classifier.joblib.pkl')
getCombinedProbs("result_20_20.csv", prob_per_author_20_20, list(allAuthors_20_20), encode_to_num_20_20, getUserName(20))
doppelganger("result_20_20.csv")

prob_per_author_40_20 = getProbsThread(3, clf, X_40_20, y_40_20, allAuthors_40_20, 'models/40_20/', '100-w10-classifier.joblib.pkl')
getCombinedProbs("result_40_20.csv", prob_per_author_40_20, list(allAuthors_40_20), encode_to_num_40_20, getUserName(40))
doppelganger("result_40_20.csv")

prob_per_author_60_20 = getProbsThread(3, clf, X_60_20, y_60_20, allAuthors_60_20, 'models/60_20/', '100-w10-classifier.joblib.pkl')
getCombinedProbs("result_60_20.csv", prob_per_author_60_20, list(allAuthors_60_20), encode_to_num_60_20, getUserName(60))
doppelganger("result_60_20.csv")

prob_per_author_60_10 = getProbsThread(3, clf, X_60_10, y_60_10, allAuthors_60_10, 'models/60_10/', '100-w10-classifier.joblib.pkl')
getCombinedProbs("result_60_10.csv", prob_per_author_60_10, list(allAuthors_60_10), encode_to_num_60_10, getUserName(60))
doppelganger("result_60_10.csv")

prob_per_author_60_30 = getProbsThread(3, clf, X_60_30, y_60_30, allAuthors_60_30, 'models/60_30/', '100-w10-classifier.joblib.pkl')
getCombinedProbs("result_60_30.csv", prob_per_author_60_30, list(allAuthors_60_30), encode_to_num_60_30, getUserName(60))
doppelganger("result_60_30.csv")


# X = getPca(selectData(40, 30))
# y = getLabel(40, 30)
#
# clf = svm.SVC(kernel='linear')
# cross = cross_val_score(clf, X, y, cv=3)
# print("cross_val_score: ", cross)
