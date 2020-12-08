import nltk
from nltk.corpus import stopwords
import pandas as pd
import spacy

#Download the package from nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

#Load the german language from spacy
nlp = spacy.load('de_core_news_sm')
#read the csv file
data = pd.read_csv("Comments.csv")
#filter the content that we want to process
info = data["content"]
#set german language and package from nltk
stop = stopwords.words('german')

result = pd.DataFrame(info)

#Create a table comparison for the raw data and the converted one
result['Content without stopwords'] = result['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
result['Lemmatized content'] = result['content'].apply(lambda x: ' '.join([token.lemma_ for token in nlp.tokenizer(x)]))
print(result)
