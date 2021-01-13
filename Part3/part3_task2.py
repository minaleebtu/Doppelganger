import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import LeaveOneGroupOut
import joblib
from joblib import Parallel, delayed
from sklearn import svm
from part3_task1 import DATA_PCA, new

authorList = new["username"].values.tolist()
datfr = pd.DataFrame(DATA_PCA)

# Separating the dependant and independent variable
X = datfr.to_numpy()
y = new['Label'].to_numpy()

# Create the probability of each author
prob_per_author = [[0]*(len(y)) for i in range(len(y))]

# Convert the dataframe into a dictionary to get the values and keys
authors_to_num = pd.Series(new["Label"].values,index=new.username).to_dict()
print(authors_to_num)

encode_to_num = pd.Series(new["Label"].values, new["Label"].values).to_dict()
print('Total authors: ', len(encode_to_num.keys()))
print('Authors are: ', encode_to_num.values())

allAuthors = encode_to_num.keys()
# Define the classifiers
clf = svm.SVC(kernel='linear')


def getLabelOfAuthor(authorname):
    authors = []
    authorsrsdict = {}
    for index in range(len(authorList)):
        authors.append(authorList[index])

    newUsers = list(set(authors))

    le = LabelEncoder()
    encoded_authors = le.fit_transform(newUsers)
    original_authors = le.inverse_transform(encoded_authors)

    for ea, oa in zip(encoded_authors, original_authors):
        authorsrsdict.update({oa: ea})
    result = None
    for author, label in authorsrsdict.items():
        if authorname == author:
            result = label
    return result


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


# Create a function for calculating the combination of all probabilities
def getCombinedProbs(selection, a, b, prob_per_author):
    # get label values of author a and b
    labelA = getLabelOfAuthor(a)
    labelB = getLabelOfAuthor(b)

    # combine probabilities (multiplication, average, squared average)
    multiple = prob_per_author[labelA][labelB] * prob_per_author[labelB][labelA]
    average = (prob_per_author[labelA][labelB] + prob_per_author[labelB][labelA]) / 2
    squared = (prob_per_author[labelA][labelB] * prob_per_author[labelA][labelB] + prob_per_author[labelB][labelA] * prob_per_author[labelB][labelA]) / 2

    # pint out probabilities(Pr(author A -> author B), Pr(author B -> author A))
    print("Pr(A->B): ", str(prob_per_author[labelA][labelB]), "& Pr(B->A): ", str(prob_per_author[labelB][labelA]))

    # return combined probability according to the way of combining probabilities
    if selection == 'multiple':
        return multiple
    elif selection == 'average':
        return average
    elif selection == 'squared':
        return squared

allAuthorNames = authors_to_num.keys()
print('Valid Author List : ',*list(allAuthorNames), sep = "\n")
prob_per_author = getProbsThread(4, clf, DATA_PCA, y, allAuthors, 'models/', '100-w10-classifier.joblib.pkl')

# get author A as an input to compare with author B
while True:
    authorA = input("Please enter the name of author A from 'Valid Author List': ")

    # if the input author is not in author list, make user input the author name again
    if getLabelOfAuthor(authorA) == None:
        print("Please enter the valid name of author")
    else:
        break

# get author B as an input to compare with author A
while True:
    authorB = input("Please enter the name of author B from 'Valid Author List': ")

    # if the input author is not in author list, make user input the author name again
    if getLabelOfAuthor(authorB) == None:
        print("Please enter the valid name of author")
    else:
        break


# get the selection of the way of combining the probabilities as an input
while True:
    selection = input("Please enter way to combine probabilities('multiple' for multiplication, 'average' for average 'squared' for squared average): ")

    # if input selection is not one of multiplication, average, squared average, make user input the selection again
    if selection.lower() not in ('multiple', 'average', 'squared'):
        print("Not an appropriate choice. Enter valid value ('multiple', 'average', 'squared')")
    else:
        break

# get the value of combined probability
combinedProb = getCombinedProbs(selection, authorA, authorB, prob_per_author)

# get threshold as an input
while True:
    threshold = input("Please enter the threshold (range: 0-1): ")

    # check the input parameter is number or not. If parameter is not number, make user input threshold again
    try:
        threshold = int(threshold)
        break
    except ValueError:
        try:
            threshold = float(threshold)
            break
        except ValueError:
            print("This is not a number. Please enter a valid number")

print("Combined probability is ", combinedProb)

# if combined probability is greater than the threshold, two authors A and B should be considered the same (Doppelgänger)
if combinedProb > threshold:
    print(">> They are Doppelgängers")
else:
    print(">> They are not Doppelgängers")
