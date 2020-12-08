import nltk
from nltk.corpus import stopwords
import pandas as pd
import spacy

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

data = pd.read_csv("Comments.csv")
info = data["content"]
result = pd.DataFrame(info)

lang = input("Enter your langauge (en for English, de for German): ")

if lang == 'en':
    # python -m spacy download en_core_web_sm
    nlp = spacy.load('en_core_web_sm')

    stop = stopwords.words('english')

    result['Content without stopwords'] = result['content'].apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))
    result['Lemmatized content'] = result['content'].apply(
        lambda x: ' '.join([token.lemma_ for token in nlp.tokenizer(x)]))
    print(result)

    result.to_csv("Pre-processing.csv", header=['Content', 'Content without stopwords', 'Lemmatized content'],
                  index=False)
elif lang == 'de':
    # python -m spacy download de_core_news_sm
    nlp = spacy.load('de_core_news_sm')

    stop = stopwords.words('german')

    result['Content without stopwords'] = result['content'].apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))
    result['Lemmatized content'] = result['content'].apply(
        lambda x: ' '.join([token.lemma_ for token in nlp.tokenizer(x)]))
    print(result)

    result.to_csv("Pre-processing.csv", header=['Content', 'Content without stopwords', 'Lemmatized content'],
                  index=False)
else:
    print("Unrecognised argument. Please specify 'en' or 'de'")
