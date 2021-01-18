from scipy.spatial.distance import euclidean
import pandas as pd


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


getDoppelLabel("result_20_20.csv")
getDoppelLabel("result_40_20.csv")
getDoppelLabel("result_60_20.csv")
getDoppelLabel("result_60_10.csv")
getDoppelLabel("result_60_30.csv")
