import pandas as pd

new = pd.read_csv('resultss.csv')
pd.options.display.width = 0

pd.options.display.float_format = "{:,.3f}".format

rtr = new.drop([ 'Encode 1', 'Encode 2'], axis = 1)
rtr['Threshold'] = rtr.mean(axis=1)
# result.drop([''])
# rtr['Sum_Of_Prob'] = rtr.apply(lambda row: row.Multiplication+ row.Averaged+row.Squared, axis=1)
# rtr['Doppelgangers'] = rtr['Sum_Of_Prob'].apply(lambda x: '0' if x <= 0.03 else '1')


print(rtr)


# print(new)
# print(os.getcwd())
