import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.model_selection import LeaveOneGroupOut
import joblib
from joblib import Parallel, delayed
from sklearn import svm

# Load the new sorted dataset
new = pd.read_csv('Joined_1.csv')
userList = new["username"].values.tolist()

# drop the unnecessary columns
datdrop = new.drop(['Unnamed: 0','Unnamed: 0.1' ,'Unnamed: 0.1.1','username', 'content', 'Label'], axis = 1)
print(datdrop)

# Just wanna see the value of each authors
new_value = new['username'].value_counts()
print(new_value)

# Convert the dataset into an array
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
dat_array = datdrop.to_numpy()
print(dat_array)

# Standardizing the data
scaler = StandardScaler()
dat_standardized = scaler.fit_transform(dat_array)
print(dat_standardized)

# Calculate the Covariance Matrix
COVMAT = np.cov(dat_standardized.T)
print(COVMAT)

# Calculate the Eigenvalues and Eigenvectors
eigvals, eigvecs = np.linalg.eig(COVMAT)

print("Eigenvalues : \n", eigvals)
print("Eigenvectors : \n", eigvecs)


# Calculate the PCA and extract the features
pd.options.display.width = 0
pca = PCA(n_components = 0.999)
DATA_PCA = pca.fit_transform(dat_array)
print("PCA Data : \n", DATA_PCA)

# See the difference shape/dimension before and after extraction
print("Shape Transformation : \n", "\nBefore Extraction :", dat_array.shape, "\nAfter Extraction Using PCA : ", DATA_PCA.shape)

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
print ('Total authors: ', len(encode_to_num.keys()))
print ('Authors are: ', encode_to_num.values())

allAuthors = encode_to_num.keys()
# Define the classifiers
# clf = LogisticRegression(penalty ='l2')
clf = svm.SVC(kernel='linear')


def getLabelOfUsers(username):
    users = []
    usersdict = {}
    for index in range(len(userList)):
        users.append(userList[index])

    newUsers = list(set(users))

    le = LabelEncoder()
    encoded_users = le.fit_transform(newUsers)
    original_users = le.inverse_transform(encoded_users)

    for eu, ou in zip(encoded_users, original_users):
        usersdict.update({ou:eu})
    result = None
    for user, label in usersdict.items():
        if username == user:
            result = label
    return result


# Define a function of
def getProbsGSThread(nthread, clf, data, label, allAuthors, modeldir, saveModel):
    crossval = LeaveOneGroupOut()

    crossval.get_n_splits(groups=label)

    prob_per_author = [[0] * (len(allAuthors)) for i in range(len(allAuthors))]
    # print(prob_per_author)

    scores = Parallel(n_jobs=nthread)(
        delayed(getProbsTrainTest)(clf, data, label, train, test, modeldir, saveModel) for train, test in
        crossval.split(data, label, groups=label))

    # print(np.array(scores).shape)
    for train, test in crossval.split(data, label, groups=label):

        anAuthor = int(label[test[0]])
        # print (anAuthor)
        train_data_label = label[train]
        trainAuthors = list(set(train_data_label))
        test_data_label = label[test]
        nTestDoc = len(scores)  # len(test_data_label)
        # print(nTestDoc)
        for j in range(nTestDoc):
            for i in range(len(trainAuthors)):
                # code.interact(local=dict(globals(), **locals()))
                try:
                    prob_per_author[anAuthor][int(trainAuthors[i])] += scores[anAuthor - 1][j][i]
                except IndexError:
                    continue
                # x[i+1] = x[i] + ( t[i+1] - t[i] ) * f( x[i], t[i] ) by x.append(x[i] + ( t[i+1] - t[i] ) * f( x[i], t[i] ))

        for i in range(len(trainAuthors)):
            prob_per_author[anAuthor][int(trainAuthors[i])] /= nTestDoc
            # code.interact(local=dict(globals(), **locals()))
    return prob_per_author


'''
       Calculate all probabilities
'''

# Create a function for calculating the probability of training and test data
def getProbsTrainTest(clf, data, label, train, test, modeldir, saveModel):
    anAuthor = int(label[test[0]])

    print("current author ", anAuthor)

    train_data = data[train, :]
    train_data_label = label[train]

    # test on anAuthor
    test_data = data[test, :]

    # check if we already have a model
    modelFile = modeldir + str(anAuthor) + "-" + saveModel

    if os.path.exists(modelFile):
        clf = joblib.load(modelFile)
    else:
        # use the following two lines if you want to choose the regularization parameters using grid search
        # parameters = {'C':[1, 10]}
        # clf = grid_search.GridSearchCV(clf, parameters)

        # train
        clf.fit(train_data, train_data_label)

        # save model
        joblib.dump(clf, modelFile, compress=9)
        print("model saved: ", modelFile)

    # get probabilities
    scores = clf.predict_proba(test_data)
    return scores


# Create a function for calculating the combination of all probabilities
def getCombinedProbs(outfile, selection, a, b, prob_per_author):
    # mul_prob = {}
    # avg_prob = {}
    # sq_prob = {}

    with open(outfile, "w+") as out:
        out.write(
            'Author 1, Author 2, P(A->B), P(B->A),Multiplication P(1->2)*P(2->1), Averaged (P(1->2)+P(2->1))/2, Squared (P(1->2)^2+P(2->1)^2)/2, Encode 1, Encode 2\n')
        # for i in range(len(allAuthors)):
        #
        #     a = int(allAuthors[i])
        #     # if len(authors_to_numbers[a])==0:
        #     #    continue
        #     for j in range(i + 1, len(allAuthors)):
        #         b = allAuthors[j]
        labelA = getLabelOfUsers(a)
        labelB = getLabelOfUsers(b)
        result = 0

        multiple = prob_per_author[labelA][labelB] * prob_per_author[labelB][labelA]
        average = (prob_per_author[labelA][labelB] + prob_per_author[labelB][labelA]) / 2
        squared = (prob_per_author[labelA][labelB] * prob_per_author[labelA][labelB] + prob_per_author[labelB][labelA] * prob_per_author[labelB][labelA]) / 2

        out.write(a + " ," + b + " ," +
                  str(prob_per_author[labelA][labelB]) + "," + str(prob_per_author[labelB][labelA]) + "," +
                  str(multiple) + "," + str(average) + "," + str(squared) + "," +
                  str(labelA) + " ," + str(labelB) +
                  "\n")

        # out.write(str(encode_to_num[a])+" ,"+str(encode_to_num[b])+" ,"+str(prob_per_author[a][b])+","+str(prob_per_author[b][a])+","+str(multiple)+","+str(average)+","+str(squared)+" ,"+str(authors_name[a])+" ,"+str(authors_name[b])+"\n")

    # if multiple in mul_prob.keys():
    #     mul_prob[multiple] += result
    # else:
    #     mul_prob[multiple] = result
    #
    # if average in avg_prob.keys():
    #     avg_prob[average] += result
    # else:
    #     avg_prob[average] = result
    # if squared in sq_prob.keys():
    #     sq_prob[squared] += result
    # else:
    #     sq_prob[squared] = result

    out.close()

    if selection == 'multiple':
        return multiple
    elif selection == 'average':
        return average
    elif selection == 'squared':
        return squared
    # return total_prob, add_prob, sq_prob


prob_per_author = getProbsGSThread(4, clf, DATA_PCA, y, allAuthors, 'models/', '100-w10-classifier.joblib.pkl')

allAuthorNames = authors_to_num.keys()


while True:
    userA = input("Please enter the name of author A: ")
    if getLabelOfUsers(userA) == None:
        print("Please enter the valid name of author")
    else:
        break

while True:
    userB = input("Please enter the name of author B: ")

    if getLabelOfUsers(userB) == None:
        print("Please enter the valid name of author")
    else:
        break


while True:
    selection = input("Please enter way to combine probabilities('multiple' for multiplication, 'average' for average 'squared' for squared average): ")

    if selection.lower() not in ('multiple', 'average', 'squared'):
        print("Not an appropriate choice. Enter valid value ('multiple', 'average', 'squared')")
    else:
        break

combined = getCombinedProbs("results.csv", selection, userA, userB, prob_per_author)

while True:
    threshold = input("Please enter the threshold (range: 0-1): ")
    try:
        threshold = int(threshold)
        break
    except ValueError:
        try:
            threshold = float(threshold)
            break
        except ValueError:
            print("This is not a number. Please enter a valid number")

# combinedValue = float(list(combined)[0])
print("Combined probability is ", combined)

if combined > threshold:
    print(">> They are Doppelgangers")
else:
    print(">> They are not Doppelgangers")
