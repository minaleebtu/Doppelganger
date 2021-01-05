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
# data = pd.read_csv("Comments.csv")
#
# df = pd.DataFrame([x for x in zip(data["username"], data["content"])], columns=['username', 'comment'])
# # print(df)
# dataList = df.reset_index(drop=True).values.tolist()

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
    print("usersdict: ", usersdict)
    for user, label in usersdict.items():
        if username == user:
            result = label
            print("username en: ", result)

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
def getCombinedProbs(selection, a, b, prob_per_author, allAuthors, encode_to_num, authors_name):
    total_prob = {}
    add_prob = {}
    sq_prob = {}

    # with open(outfile, "w+") as out:
    #     out.write(
    #         'Author 1, Author 2, P(A->B), P(B->A),Multiplication P(1->2)*P(2->1), Averaged (P(1->2)+P(2->1))/2, Squared (P(1->2)^2+P(2->1)^2)/2, Encode 1, Encode 2\n')
    #     for i in range(len(allAuthors)):
    #
    #         a = int(allAuthors[i])
    #         # if len(authors_to_numbers[a])==0:
    #         #    continue
    #         for j in range(i + 1, len(allAuthors)):
    #             b = allAuthors[j]
    #
    result = 0

    total = prob_per_author[a][b] * prob_per_author[b][a]
    addition = (prob_per_author[a][b] + prob_per_author[b][a]) / 2
    sqsum = (prob_per_author[a][b] * prob_per_author[a][b] + prob_per_author[b][a] * prob_per_author[b][
        a]) / 2
    #
    #             out.write(str(authors_name[a]) + " ," + str(authors_name[b]) + " ," +
    #                       str(prob_per_author[a][b]) + "," + str(prob_per_author[b][a]) + "," +
    #                       str(total) + "," + str(addition) + "," + str(sqsum) + "," +
    #                       str(encode_to_num[a]) + " ," + str(encode_to_num[b]) +
    #                       "\n")
    #
    #             # out.write(str(encode_to_num[a])+" ,"+str(encode_to_num[b])+" ,"+str(prob_per_author[a][b])+","+str(prob_per_author[b][a])+","+str(total)+","+str(addition)+","+str(sqsum)+" ,"+str(authors_name[a])+" ,"+str(authors_name[b])+"\n")
    #
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
    #
    # out.close()
    if selection == 'multiple':
        return total_prob
    elif selection == 'average':
        return add_prob
    elif selection == 'squared':
        return sq_prob
    # return total_prob, add_prob, sq_prob


prob_per_author = getProbsGSThread(4, clf, DATA_PCA, y, allAuthors, 'models/', '100-w10-classifier.joblib.pkl')

allAuthorNames = authors_to_num.keys()

while True:
    selection = input("Please enter way to combine probabilities('multiple' for multiplication, 'average' for average 'squared' for squared average): ")

    if selection.lower() not in ('multiple', 'average', 'squared'):
        print("Not an appropriate choice. Enter valid value ('multiple', 'average', 'squared')")
    else:
        break

a = {'Super-Duper Missile': ' "Wenn Trump Jr. in die Schlagzeilen ger??t, weil er in der Mongolei ein unter besonderem Schutz stehendes Schaf erlegt, ohne eine Genehmigung daf??r zu haben, erheitert das viele Trump-Anh??nger sicherlich eher, als dass es sie abschreckt. " Und dieses asoziale Gehabe seiner Basis ist das eigentlich gef??hrliche was immer wieder in verschiedensten Situationen offensichtlich wird. Es geht einzig und allein um das eigene Weltbild. Verst??ndnis f??r andere, R??cksicht f??r andere - alles egal, hauptsache man kann den eigenen "Lebensstil" weiter leben. Mmn ist fehlende Bildung das gr????te Problem. Aber es ist durchaus so gewollt von der "eigenen" Partei. '}
b = {'aaaaaaaaaaaaaaaasssssssssssssdddddddddd': ' "Wenn Trump Jr. in die Schlagzeilen ger??t, weil er in der Mongolei ein unter besonderem Schutz stehendes Schaf erlegt, ohne eine Genehmigung daf??r zu haben, erheitert das viele Trump-Anh??nger sicherlich eher, als dass es sie abschreckt. " Und dieses asoziale Gehabe seiner Basis ist das eigentlich gef??hrliche was immer wieder in verschiedensten Situationen offensichtlich wird. Es geht einzig und allein um das eigene Weltbild. Verst??ndnis f??r andere, R??cksicht f??r andere - alles egal, hauptsache man kann den eigenen "Lebensstil" weiter leben. Mmn ist fehlende Bildung das gr????te Problem. Aber es ist durchaus so gewollt von der "eigenen" Partei. '}
# b = {'whitemouse': ' "Auch um Kraft zu sammeln, die wir brauchen um uns gegen die zu wehren, denen an der glimpflichen Bew??ltigung der Pandemie wenig gelegen ist. Aus welchen Gr??nden auch immer." Das wird schwierig,  wenn die immer wieder von Verwaltungsrichtern R??ckendeckung erhalten. In K??ln d??rfen die Menschen (zu Recht) nicht auf der Stra??e Karneval feiern, aber Wirrdenker d??rfen sich auf der Stra??e versammeln. W??hrend Karnevalisten sich noch bem??hen w??rden, die Abstands- und Maskenregeln einzuhalten,   bem??hen sich die Wirrdenker, ein m??glichst virenverbreitendes Verhalten an den Tag zu legen. Was daran sch??tzenswert sein soll, erschlie??t sich mir nicht. Und den Schwachsinn, den diese Leute von sich geben, hat nun mittlerweile jeder, der lesen  h??ren oder sehen kann, zur Kenntnis genommen. Ebenso, dass es einen gewissen Bodensatz der Gesellschaft gibt, der ihn vertritt. '}
userA = ''.join(list(a.keys()))
userB = ''.join(list(b.keys()))
print("userA: ", userA)
print("userB: ", userB)
print("A: ", getLabelOfUsers(userA))
print("A: ", getLabelOfUsers(userB))

combined = getCombinedProbs(selection, getLabelOfUsers(userA), getLabelOfUsers(userB), prob_per_author, list(allAuthors), encode_to_num, list(allAuthorNames))

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

combinedValue = float(list(combined)[0])

print("combined: ", type(combinedValue), combinedValue, combined)
print("threshold: ", type(threshold), threshold)

if combinedValue > threshold:
    print("They are Doppelgangers")
else:
    print("They are not Doppelgangers")