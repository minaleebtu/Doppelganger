from __future__ import print_function, unicode_literals
import nltk
from nltk.corpus import stopwords
import pandas as pd
import spacy

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

nlp = spacy.load('de_core_news_sm')
data = pd.read_csv("Comments.csv")

info = data["content"]
stop = stopwords.words('german')

result = pd.DataFrame(info)

result['Content without stopwords'] = result['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
result['Lemmatized content'] = result['content'].apply(lambda x: ' '.join([token.lemma_ for token in nlp.tokenizer(x)]))
print(result)

result.to_csv("Pre-processing.csv", header=['Content', 'Content without stopwords', 'Lemmatized content'], index=False)
