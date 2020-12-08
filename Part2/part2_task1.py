import nltk
from nltk.corpus import stopwords
import pandas as pd
import spacy

#download the package from nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')

#read the csv file and open the specific column
data = pd.read_csv("Comments.csv")
info = data["content"]
result = pd.DataFrame(info)

#create an input to choose the language
lang = input("Enter your langauge (en for English, de for German): ")

#create a condition for english and german language
if lang == 'en':
    # python -m spacy download en_core_web_sm
    # load the package for english language for lemmatizing
    nlp = spacy.load('en_core_web_sm')

    # choose the english language for stopwords
    stop = stopwords.words('english')

    #create a table comparison from the datasets for removeing stopwords and lemmatizing

    result['Content without stopwords'] = result['content'].apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))
    result['Lemmatized content'] = result['content'].apply(
        lambda x: ' '.join([token.lemma_ for token in nlp.tokenizer(x)]))
    print(result)

    result.to_csv("Pre-processing.csv", header=['Content', 'Content without stopwords', 'Lemmatized content'],
                  index=False)
elif lang == 'de':
    # python -m spacy download de_core_news_sm

    #load the package for german language for lemmatizing
    nlp = spacy.load('de_core_news_sm')

    #choose the german language for stopwords
    stop = stopwords.words('german')

    #create a table comparison from the datasets for removeing stopwords and lemmatizing
    result['Content without stopwords'] = result['content'].apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))
    result['Lemmatized content'] = result['content'].apply(
        lambda x: ' '.join([token.lemma_ for token in nlp.tokenizer(x)]))
    print(result)

    #write the result into csv file
    result.to_csv("Pre-processing.csv", header=['Content', 'Content without stopwords', 'Lemmatized content'],
                  index=False)

    #If none of the language is choosen it will print "unrecognised argument"
else:
    print("Unrecognised argument. Please specify 'en' or 'de'")
