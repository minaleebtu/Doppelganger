from scipy.spatial.distance import euclidean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Create a list for the classification score
Doppelganger = []
Doppelganger1 = []
Euclidean = []
Euclidean1 = []

# Function for calculating with Euclidean Distance
def getDoppelLabel(outfile):
    dataset = pd.read_csv(outfile)
    probAB = dataset['P(A->B)']
    probBA = dataset['P(B->A)']
    # print(probAB, probBA)
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    probAB_arr = probAB.to_numpy()
    probBA_arr = probBA.to_numpy()

    # get threshold as an input
    while True:
        threshold = input("[" + outfile + "] Please enter the threshold (should be number): ")

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

    distList = []
    for a, b in zip(probAB_arr, probBA_arr):
        dist = euclidean(a, b)
        distList.append(dist)
        print("dist: ", dist)
    dataset['Euclidean Distance'] = distList
    dataset['Euclidean Doppelgangers'] = dataset['Euclidean Distance'].apply(lambda x: 1 if x < threshold else 0)

    return dataset.to_csv(outfile, index=False)

# Function for calculating the Euclidean score
def predict_score(df):
    data = pd.read_csv(df)
    X = data.drop(['Doppelgangers', 'Author 1', 'Author 2', 'Euclidean Doppelgangers'], axis = 1)
    y = data['Euclidean Doppelgangers']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=40)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    return score

#Function for calculating Doppelganger score
def doppelganger_score(df):
    data = pd.read_csv(df)
    X = data.drop(['Doppelgangers', 'Author 1', 'Author 2', 'Euclidean Doppelgangers'], axis = 1)
    y = data['Euclidean Doppelgangers']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=40)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    return score


# getDoppelLabel("result_20_20.csv")
# getDoppelLabel("result_40_20.csv")
# getDoppelLabel("result_60_20.csv")
# getDoppelLabel("result_60_10.csv")
# getDoppelLabel("result_60_30.csv")

print('Euclidean Classification Score With 20 Users & 20 Comments :\n',predict_score('result_20_20.csv'))
print('Euclidean Classification Score With 40 Users & 20 Comments :\n',predict_score('result_40_20.csv'))
print('Euclidean Classification Score With 60 Users & 20 Comments :\n',predict_score('result_60_20.csv'))
print('Euclidean Classification Score With 60 Users & 10 Comments :\n',predict_score('result_60_10.csv'))
print('Euclidean Classification Score With 60 Users & 20 Comments :\n',predict_score('result_60_20.csv'))
print('Euclidean Classification Score With 30 Users & 30 Comments :\n',predict_score('result_60_30.csv'))
print('-'*80)
print('Doppelganger Classification Score With 20 Users & 20 Comments :', doppelganger_score('result_20_20.csv'))
print('Euclidean Classification Score With 20 Users & 20 Comments :',predict_score('result_20_20.csv'))
print('-'*80)
print('\nDoppelganger Classification Score With 40 Users & 20 Comments :', doppelganger_score('result_40_20.csv'))
print('Euclidean Classification Score With 40 Users & 20 Comments :',predict_score('result_40_20.csv'))
print('-'*80)
print('\nDoppelganger Classification Score With 60 Users & 20 Comments :', doppelganger_score('result_60_20.csv'))
print('Euclidean Classification Score With 60 Users & 20 Comments :',predict_score('result_60_20.csv'))
print('-'*80)
print('\nDoppelganger Classification Score With 60 Users & 10 Comments :', doppelganger_score('result_60_10.csv'))
print('Euclidean Classification Score With 60 Users & 10 Comments :',predict_score('result_60_10.csv'))
print('-'*80)
print('\nDoppelganger Classification Score With 60 Users & 20 Comments :', doppelganger_score('result_60_20.csv'))
print('Euclidean Classification Score With 60 Users & 20 Comments :',predict_score('result_60_20.csv'))
print('-'*80)
print('\nDoppelganger Classification Score With 60 Users & 30 Comments :', doppelganger_score('result_60_30.csv'))
print('Euclidean Classification Score With 60 Users & 30 Comments :',predict_score('result_60_30.csv'))


# Append the scores to the doppelganger list (User per Comment)
Doppelganger.append(doppelganger_score('result_20_20.csv'))
Doppelganger.append(doppelganger_score('result_40_20.csv'))
Doppelganger.append(doppelganger_score('result_60_20.csv'))

# Append the scores to the doppelganger list (Comment per User)
Doppelganger1.append(doppelganger_score('result_60_10.csv'))
Doppelganger1.append(doppelganger_score('result_60_20.csv'))
Doppelganger1.append(doppelganger_score('result_60_30.csv'))

# Append the scores to the Euclidean list (User per Comment)
Euclidean.append(predict_score('result_20_20.csv'))
Euclidean.append(predict_score('result_40_20.csv'))
Euclidean.append(predict_score('result_60_20.csv'))

# Append the scores to the Euclidean list (Comment per User)
Euclidean1.append(predict_score('result_60_10.csv'))
Euclidean1.append(predict_score('result_60_20.csv'))
Euclidean1.append(predict_score('result_60_30.csv'))

# Create a plot for the classifier's score
x = [1,2,3]

plt.plot(x, Doppelganger, label='Pseudonyms of Doppelganger')
plt.plot(x, Doppelganger1, label='Comments of Doppelganger')
plt.plot(x, Euclidean, label='Pseudonyms of Euclidean')
plt.plot(x, Euclidean1, label='Comments of Euclidean')
plt.xlabel('Number of Experiments')
plt.ylabel('Value of Classifier')

plt.title("Classifier Result")
plt.legend()

plt.show()
