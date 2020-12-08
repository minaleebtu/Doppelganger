import re
import nltk
from nltk.corpus import stopwords
from textblob_de import TextBlobDE as TextBlob

def cleantext(text):
    low_case = str(text).lower()
    ext = re.sub(r'[^A-Za-zäöüÄÖÜ0-9]+', " ", str(low_case))
    return ext

def stop_words(text):
    bb = cleantext(text)
    stop = stopwords.words('german')
    words = nltk.word_tokenize(str(bb))

    STOPWORDS = []
    for word in words:
        if word not in stop:
            STOPWORDS.append(word)
    return STOPWORDS



def lemmatize (text):

    aa = cleantext(text)
    blob = TextBlob(str(aa))

    return blob.words.lemmatize()


def word_token(text):
    cc = cleantext(text)
    words = nltk.word_tokenize(str(cc))
    return words

