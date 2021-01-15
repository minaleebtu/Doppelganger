import pandas as pd
import csv
from collections import defaultdict

# pd.options.display.max_columns = None
# pd.options.display.max_rows = None

dataset = pd.read_csv('Comments.csv')

dataset = dataset.drop('commDate', axis=1)
duplicate = dataset[dataset.duplicated()]
duplicate.to_csv("duplicate.csv", index=False)
print("duplicate:\n", duplicate)
print("before:\n", dataset.shape)
dataset = dataset.drop_duplicates()
# making a bool series
# bool_series = dataset.duplicated()
# passing NOT of bool series to see unique values only
# dataset = dataset[~bool_series]
print("after:\n", dataset.shape)
dataset.to_csv("dataset.csv", index=False)
usersInDS = dataset['username']
commentsInDS = dataset['content']
# commentsInDS.drop_duplicates(keep=False, inplace=True)
# duComm = dataset[commentsInDS.duplicated()]
# print(duComm)

userCnt = usersInDS.value_counts().to_dict()


def users_with_30comments(userCnt):
    print("len of all userCount: ", len(userCnt))
    result = {}
    for user, numOfcomm in userCnt.items():
        if numOfcomm >= 30:
            result.update({user:numOfcomm})
    print("len of result(users_with_30comments): ", len(result))

    return result


def getUserComm():
    result = []
    resultDict = {}
    # print("len of usersInDS.values", len(usersInDS.values))
    # print("len of commentsInDS.values", len(commentsInDS.values))
    for user, comm in zip(usersInDS.values, commentsInDS.values):
        result.append([user,comm])
    print("result(getUserComm) len: ", len(result))
    getUserComm = pd.DataFrame(result)
    getUserComm.to_csv("getUserComm.csv", header=['username', 'comment'], index=False)
    return result


def getComment(userCommDict):
    users = userCommDict.keys()
    result = []
    resultDict = {}
    cnt = 0
    # print("getUserComm(): ", len(getUserComm()))
    for pdUser, comm in getUserComm():
        if pdUser in users:
            result.append([pdUser, comm])
            # print("pdUser: ", pdUser, " & comm: ", comm)
            cnt += 1
            # print("user: ", user, " & pdUser: ", pdUser)
            # resultDict.update({pdUser:comm})
            # result.append(resultDict)

            # print("cnt: ", cnt)
    return result


result = getComment(users_with_30comments(userCnt))
new = pd.DataFrame(result)
# print(new)
new.to_csv("usercomm.csv", header=['username', 'comment'], index=False)
# with open('usercomm.csv', 'w', newline='', encoding='utf-8') as output_file:
#     # fieldnames = ['username', 'comment']
#     # writer = csv.DictWriter(output_file, fieldnames=fieldnames)
#     writer = csv.writer(output_file)
#     # writer.writeheader()
#     writer.writerow(('username', 'comment'))
#     for user, comment in result:
#     # for resultDict in result:
#     #     for user, comment in resultDict.items():
#         writer.writerow([user, comment])
# print(new)
usercomm = pd.read_csv('usercomm.csv')
print(usercomm)
print("user cnt:\n", usercomm['username'].value_counts())
newUserCnt = usercomm['username'].value_counts().to_dict()
print("newUserCnt: ", newUserCnt, len(newUserCnt))



def getUserCommByNum(numOfUser, numOfComm):
    group = usercomm.groupby('username')['comment'].apply(list).to_dict()
    # print("groupby:\n", group, len(group))
    result = []
    resultDict = {}
    preResult = []
    # print("keys: ", len(group.keys()[0]))
    # print("values: ", len(group.values()[0]))
    # print("enumerate(group.items()): ", enumerate(group.items()))
    listGroup = [(k,v) for k,v in group.items()]
    # print("listGroup: ", listGroup[0][1][1311])
    for userIn in range(0, numOfUser):
        # print("listGroup[", userIn, "]: ", listGroup[userIn])
        preResult.append(listGroup[userIn])
    # print("preResult: ", preResult[1][0])
    for pre in preResult:
        # print("pre: ", pre)
        # print("pre[0]: ", pre[0])
        # result.append(pre[0])
        for commIn in range(0, numOfComm):
            # print("pre[0]: ", pre[0])
            # print("pre[1][", commIn, "]: ", pre[1][commIn])
            if pre[0] in resultDict:
                # append the new number to the existing array at this slot
                resultDict[pre[0]].append(pre[1][commIn])
            else:
                # create a new array in this slot
                resultDict[pre[0]] = [pre[1][commIn]]
            # resultDict[pre[0]].append(pre[1][commIn])
            # resultDict.update({pre[0]:pre[1][commIn]})
            # result.append(resultDict.update({pre[0]:pre[1][commIn]}))
    print("resultDict: ", resultDict, len(resultDict))
    for k,v in resultDict.items():
        print("k: ", k, " & num of v: ", len(v))
    return resultDict

newResult = getUserCommByNum(60,30)
print(newResult, type(newResult), len(newResult))

with open('sortedData.csv', 'w', newline='', encoding='utf-8') as output_file:
    # fieldnames = ['username', 'comment']
    # writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer = csv.writer(output_file)
    # writer.writeheader()
    writer.writerow(('username', 'comment'))
    for user, comment in newResult.items():
        for comm in comment:
    # for resultDict in result:
    #     for user, comment in resultDict.items():
            writer.writerow([user, comm])
sortedData = pd.read_csv('sortedData.csv')
print("sortedData:\n", sortedData)
