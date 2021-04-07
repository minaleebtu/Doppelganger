# -*- coding: utf-8 -*-
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import textstat
import numpy as np
import scipy.stats as stats
import language_tool_python
import string
from textblob_de import TextBlobDE as TextBlob
from germansentiment import SentimentModel
import spacy
import collections
# from Cleantext import stop_words
import langdetect


def multiple_whitespace(text):
    count = text.count('  ')
    return count


def count_punctuation(text):
    return len(list(filter(lambda c: c in string.punctuation, text)))


def count_senti(list):
    cntPos = 0
    cntNeg = 0
    for text in list:
        if text == 'positive':
            cntPos += 1
        elif text == 'negative':
            cntNeg += 1

    return cntPos, cntNeg


def shannon_entropy(words):
    length = len(words)
    freqs = Counter(words)
    distribution = np.array(list(freqs.values()))
    distribution = np.true_divide(distribution, length)

    E = stats.entropy(distribution, base=2)

    return E


def preprocess(tokens):
    text_length = len(tokens)
    frequency_list = collections.Counter(tokens)
    frequency_spectrum = dict(collections.Counter(frequency_list.values()))
    return text_length, frequency_spectrum


def simpson_d(text_length, frequency_spectrum):
    return sum((freq_size * (freq / text_length) * ((freq - 1) / (text_length - 1)) for freq, freq_size in
                frequency_spectrum.items()))


def sichel_s(vocabulary_size, frequency_spectrum):
    return frequency_spectrum.get(2, 0) / vocabulary_size


def leetspeak(text):
    for char in text:
        if char == 'a':
            text = text.replace('a', '4')
        elif char == 'b':
            text = text.replace('b', '8')
        elif char == 'e':
            text = text.replace('e', '3')
        elif char == 'l':
            text = text.replace('l', '1')
        elif char == 'o':
            text = text.replace('o', '0')
        elif char == 's':
            text = text.replace('s', '5')
        elif char == 't':
            text = text.replace('t', '7')
        else:
            pass
    return text


def getFeatures(selectedData):
    textstat.set_lang("de")

    # nltk.download('averaged_perceptron_tagger')

    tool = language_tool_python.LanguageTool('de')

    nlp = spacy.load('de_core_news_sm')

    model = SentimentModel()

    data = selectedData

    contentList = data["comment"].values.tolist()

    numberOfWordsPerComm = []

    charList = []
    digitList = []
    upperList = []
    lowerList = []

    syllablesTuple = ()
    syllablesList = []

    typeTokens = []
    typeTokenRatio = []

    entropyTuple = ()
    entropyList = []

    ease_reading = []
    gunning_fog = []

    simpsonList = []
    sichelList = []

    fleschTuple = ()
    fleschList = []

    sentenceLenPerCommList = []

    grammarChkPerSenTuple = ()
    grammarChkPerSenList = []

    # grammarChkPerCommTuple = ()
    grammarChkPerCommList = []

    largeWordCountTuple = ()
    largeWordCountList = []

    upperWordPerSenTuple = ()
    upperWordPerSenList = []
    # upperWordPerCommTuple = ()
    upperWordPerCommList = []

    puncCountPerSenTuple = ()
    puncCountPerSenList = []
    # puncCountPerCommTuple = ()
    puncCountPerCommList = []

    multiSpacePerSenTuple = ()
    multiSpacePerSenList = []
    # multiSpacePerCommTuple = ()
    multiSpacePerCommList = []

    NERTuple = ()
    NERList = []

    nounPhraseList = []

    langDetectList = []

    wordCount = 0
    charCount = 0
    digitCount = 0
    upperCount = 0
    lowerCount = 0
    syllableCount = 0

    sentenceCount = 0
    shortSenCount = 0
    longSenCount = 0
    wordCountPerSentence = 0
    tot_wordCountPerSentence = 0

    puncCountPerComm = 0

    totPosPerWord = 0
    totPosPerSen = 0
    totNegPerWord = 0
    totNegPerSen = 0

    digit_reg = r'[0-9]'

    punctuation = string.punctuation

    for row in contentList:
        strip_row = row.strip()
        split_by_word = strip_row.split()
        ease_r = textstat.flesch_reading_ease(strip_row)
        ease_reading.append(ease_r)
        fog = textstat.gunning_fog(strip_row)
        gunning_fog.append(fog)
        langDetectList.append(langdetect.detect(strip_row))

        sentences = sent_tokenize(strip_row)
        sentenceCount += len(sentences)

        puncCountPerComm = count_punctuation(strip_row)
        # puncCountPerCommTuple = strip_row, puncCountPerComm
        puncCountPerCommList.append(puncCountPerComm)

        # multiSpacePerCommTuple = strip_row, multiple_whitespace(strip_row)
        multiSpacePerCommList.append(multiple_whitespace(strip_row))

        pre = preprocess(strip_row)
        text_length = pre[0]
        frequency_spectrum = pre[1]
        simpsonList.append(simpson_d(text_length, frequency_spectrum))
        sichelList.append(sichel_s(text_length, frequency_spectrum))

        sentenceLength = 0

        grammarChkCnt = 0

        largeWordCount = 0

        puncCountPerSen = 0

        for sentence in sentences:
            wordsInSen = sentence.split()
            upperCntPerSen = 0
            for eachWord in wordsInSen:
                if any(u.isupper() for u in eachWord):
                    upperCntPerSen += 1

            upperWordPerSenTuple = sentence, upperCntPerSen
            upperWordPerSenList.append(upperWordPerSenTuple)

            puncCountPerSen = count_punctuation(sentence)
            puncCountPerSenTuple = sentence, puncCountPerSen
            puncCountPerSenList.append(puncCountPerSenTuple)

            multiSpacePerSenTuple = sentence, multiple_whitespace(sentence)
            multiSpacePerSenList.append(multiSpacePerSenTuple)

            fleschkin = textstat.flesch_kincaid_grade(sentence)
            fleschTuple = sentence, fleschkin
            fleschList.append(fleschTuple)

            wordCountPerSentence = len(sentence.split())

            if wordCountPerSentence >= 50:
                longSenCount += 1
            elif wordCountPerSentence <= 20:
                shortSenCount += 1

            tot_wordCountPerSentence += wordCountPerSentence

            sentenceLength += len(sentence)

            grammarChkSen = tool.check(sentence)
            grammarChkCnt += len(grammarChkSen)

            grammarChkPerSenTuple = sentence, len(grammarChkSen)
            grammarChkPerSenList.append(grammarChkPerSenTuple)

            doc = nlp(sentence)

            for entity in doc.ents:
                if not entity.text.startswith("http"):
                    NERTuple = entity.text, entity.label_
                    NERList.append(NERTuple)

        # grammarChkPerCommTuple = strip_row, grammarChkCnt
        grammarChkPerCommList.append(grammarChkCnt)

        sentenceLenPerCommList.append(sentenceLength / len(sentences))

        numberOfWordsPerComm.append(len(split_by_word))

        blob = TextBlob(strip_row)

        nounPhrase = blob.noun_phrases
        nounPhraseList.append(nounPhrase)

        tags = blob.tags
        typeTokens.append(tags)

        upperCntPerComm = 0

        for word in split_by_word:
            if any(u.isupper() for u in word):
                upperCntPerComm += 1

            charList.append(len(re.sub(digit_reg, "", word)))
            charCount += len(re.sub(digit_reg, "", word))

            upperCountPerWord = sum(u.isupper() for u in word)
            lowerCountPerWord = sum(l.islower() for l in word)
            upperList.append(upperCountPerWord)
            lowerList.append(lowerCountPerWord)
            upperCount += upperCountPerWord
            lowerCount += lowerCountPerWord

            numbers = re.findall("\d", word)

            digitList.append(len(numbers))
            digitCount += len(numbers)

            if word.startswith("http"):
                syllableCount = 0
                syllablesTuple = word, syllableCount
                syllablesList.append(syllablesTuple)
            elif not word.isalpha():
                revisedWord = "".join(re.findall("[a-zA-ZäöüÄÖÜ]+", word))
                if len(revisedWord) >= 10:
                    largeWordCount += 1

                syllableCount = textstat.syllable_count(revisedWord)
                syllablesTuple = word, syllableCount
                syllablesList.append(syllablesTuple)
            else:
                if len(word) >= 10:
                    largeWordCount += 1

                syllableCount = textstat.syllable_count(word)
                syllablesTuple = word, syllableCount
                syllablesList.append(syllablesTuple)

            entropyTuple = word, shannon_entropy(word)
            entropyList.append(entropyTuple)

        # upperWordPerCommTuple = strip_row, upperCntPerComm
        upperWordPerCommList.append(upperCntPerComm)

        largeWordCountList.append(largeWordCount)
        wordCount += len(split_by_word)

        sentimentSen = model.predict_sentiment(sentences)
        sentiCntPerSen = count_senti(sentimentSen)
        posCntPerSen = sentiCntPerSen[0]
        negCntPerSen = sentiCntPerSen[1]

        totPosPerSen += posCntPerSen
        totNegPerSen += negCntPerSen

        sentimentWord = model.predict_sentiment(split_by_word)
        sentiCntPerWord = count_senti(sentimentWord)
        posCntPerWord = sentiCntPerWord[0]
        negCntPerWord = sentiCntPerWord[1]

        totPosPerWord += posCntPerWord
        totNegPerWord += negCntPerWord

    typeTokenCount = Counter(tag for tokens in typeTokens for word, tag in tokens)
    sbase = sum(typeTokenCount.values())

    for el, cnt in typeTokenCount.items():
        ratio = el, '{0:2.2f}%'.format((100.0 * cnt) / sbase)
        typeTokenRatio.append(ratio)

    return [numberOfWordsPerComm, largeWordCountList, simpsonList, sichelList, sentenceLenPerCommList,
            puncCountPerCommList, multiSpacePerCommList, grammarChkPerCommList, upperWordPerCommList,
            list(set(ease_reading)), list(set(gunning_fog)), sentenceLenPerCommList]
