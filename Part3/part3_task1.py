import pandas as pd
from Part2.part2_task2 import numberOfWordsPerComm,  largeWordCountList, simpsonList, sichelList, sentenceLenPerCommList, puncCountPerCommList, multiSpacePerCommList, grammarChkPerCommList, upperWordPerCommList, ease_reading, gunning_fog

print("numberOfWordsPerComm: ", numberOfWordsPerComm)
print("largeWordCountList: ", largeWordCountList)
print("simpsonList: ", simpsonList)
print("sichelList: ", sichelList)
print("sentenceLenPerCommList: ", sentenceLenPerCommList)
print("puncCountPerCommList: ", puncCountPerCommList)
print("multiSpacePerCommList: ", multiSpacePerCommList)
print("grammarChkPerCommList: ", grammarChkPerCommList)
print("upperWordPerCommList: ", upperWordPerCommList)
print("ease_reading: ", ease_reading)
print("gunning_fog: ", gunning_fog)

# df = pd.DataFrame(numberOfWordsPerComm)
df = pd.DataFrame([x for x in zip(numberOfWordsPerComm,  largeWordCountList, simpsonList, sichelList, sentenceLenPerCommList, puncCountPerCommList, multiSpacePerCommList, grammarChkPerCommList, upperWordPerCommList, ease_reading, gunning_fog)], columns=['numberOfWordsPerComm',  'largeWordCountList', 'simpsonList', 'sichelList', 'sentenceLenPerCommList', 'puncCountPerCommList', 'multiSpacePerCommList', 'grammarChkPerCommList', 'upperWordPerCommList', 'ease_reading', 'gunning_fog'])
print(df)
