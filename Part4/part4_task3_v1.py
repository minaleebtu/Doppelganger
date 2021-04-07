from scipy.spatial.distance import euclidean
import pandas as pd
import numpy as np

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


print("euclidDoppel:\n", euclidDoppel(selectData(20, 20)))

un_euclid = euclidDoppel(selectData(20, 20))['Euclidean Doppelgangers'].values.tolist()
print(len(un_euclid))