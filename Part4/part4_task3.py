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

# get threshold as an input
while True:
    threshold = input("Please enter the threshold (should be number): ")

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


# Function for calculating with Euclidean Distance and appending result to csv file
def euclidDoppel(outfile):
    dataset = pd.read_csv(outfile)

    nameOfFile_index = outfile.find('.csv')
    nameOfFile = outfile[:nameOfFile_index]

    probAB = dataset['P(A->B)']
    probBA = dataset['P(B->A)']

    probAB_arr = probAB.to_numpy()
    probBA_arr = probBA.to_numpy()

    distList = []

    # Calculating Euclidean distance between two users of all users
    for a, b in zip(probAB_arr, probBA_arr):
        dist = euclidean(a, b)
        distList.append(dist)

    dataset['Euclidean Distance'] = distList
    dataset['Euclidean Doppelgangers'] = dataset['Euclidean Distance'].apply(lambda d: 1 if d < threshold else 0)

    return dataset.to_csv(nameOfFile+"_incl_euclid.csv", index=False)


# Function for calculating the Euclidean score
def predict_score(df):
    data = pd.read_csv(df)

    X = data.drop(['Doppelgangers', 'Author 1', 'Author 2', 'Euclidean Doppelgangers'], axis=1)
    y = data['Euclidean Doppelgangers']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=40)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    return score


#Function for calculating Doppelganger score
def doppelganger_score(df):
    data = pd.read_csv(df)

    X = data.drop(['Doppelgangers', 'Author 1', 'Author 2'], axis=1)
    y = data['Doppelgangers']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = RandomForestClassifier(n_estimators=40)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    return score


# Appending Euclidean distance values (Unsupervised)
euclidDoppel("result_20_20.csv")
euclidDoppel("result_40_20.csv")
euclidDoppel("result_60_20.csv")
euclidDoppel("result_60_10.csv")
euclidDoppel("result_60_30.csv")

print('-'*80)
print('Doppelganger Classification Score With 20 Users & 20 Comments :', doppelganger_score('result_20_20.csv'))
print('Euclidean Classification Score With 20 Users & 20 Comments :', predict_score('result_20_20_incl_euclid.csv'))
print('-'*80)

print('\nDoppelganger Classification Score With 40 Users & 20 Comments :', doppelganger_score('result_40_20.csv'))
print('Euclidean Classification Score With 40 Users & 20 Comments :', predict_score('result_40_20_incl_euclid.csv'))
print('-'*80)

print('\nDoppelganger Classification Score With 60 Users & 20 Comments :', doppelganger_score('result_60_20.csv'))
print('Euclidean Classification Score With 60 Users & 20 Comments :', predict_score('result_60_20_incl_euclid.csv'))
print('-'*80)

print('\nDoppelganger Classification Score With 60 Users & 10 Comments :', doppelganger_score('result_60_10.csv'))
print('Euclidean Classification Score With 60 Users & 10 Comments :', predict_score('result_60_10_incl_euclid.csv'))
print('-'*80)

print('\nDoppelganger Classification Score With 60 Users & 20 Comments :', doppelganger_score('result_60_20.csv'))
print('Euclidean Classification Score With 60 Users & 20 Comments :', predict_score('result_60_20_incl_euclid.csv'))
print('-'*80)

print('\nDoppelganger Classification Score With 60 Users & 30 Comments :', doppelganger_score('result_60_30.csv'))
print('Euclidean Classification Score With 60 Users & 30 Comments :', predict_score('result_60_30_incl_euclid.csv'))


# Appending the scores to the doppelganger list (User per Comment)
Doppelganger.append(doppelganger_score('result_20_20.csv'))
Doppelganger.append(doppelganger_score('result_40_20.csv'))
Doppelganger.append(doppelganger_score('result_60_20.csv'))

# Appending the scores to the doppelganger list (Comment per User)
Doppelganger1.append(doppelganger_score('result_60_10.csv'))
Doppelganger1.append(doppelganger_score('result_60_20.csv'))
Doppelganger1.append(doppelganger_score('result_60_30.csv'))

# Appending the scores to the Euclidean list (User per Comment)
Euclidean.append(predict_score('result_20_20_incl_euclid.csv'))
Euclidean.append(predict_score('result_40_20_incl_euclid.csv'))
Euclidean.append(predict_score('result_60_20_incl_euclid.csv'))

# Appending the scores to the Euclidean list (Comment per User)
Euclidean1.append(predict_score('result_60_10_incl_euclid.csv'))
Euclidean1.append(predict_score('result_60_20_incl_euclid.csv'))
Euclidean1.append(predict_score('result_60_30_incl_euclid.csv'))

# Creating a plot for the classifier's score
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
