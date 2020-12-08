# -*- coding: utf-8 -*-
import sys
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
from textblob import TextBlob
from germansentiment import SentimentModel
import spacy
import collections
from Cleantext import stop_words
import langdetect

# sys.stdout = open('result.txt', 'w', encoding='utf-8')

textstat.set_lang("de")

nltk.download('averaged_perceptron_tagger')

tool = language_tool_python.LanguageTool('de')

nlp = spacy.load('de_core_news_sm')
# nlp = spacy.load('de_core_news_md')

model = SentimentModel()

data = pd.read_csv("Comments.csv")

contentList = data["content"].values.tolist()

numberOfWordsPerComm = []

tokenList = []

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

grammarChkPerCommTuple = ()
grammarChkPerCommList = []

largeWordCountTuple = ()
largeWordCountList = []

upperWordPerSenTuple = ()
upperWordPerSenList = []
upperWordPerCommTuple = ()
upperWordPerCommList = []

puncCountPerSenTuple = ()
puncCountPerSenList = []
puncCountPerCommTuple = ()
puncCountPerCommList = []

multiSpacePerSenTuple = ()
multiSpacePerSenList = []
multiSpacePerCommTuple = ()
multiSpacePerCommList = []

NERTuple = ()
NERList = []

nounPhraseList = []

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

sample = [' Die  europäischen   Regierungen stolpern von einer sinnlosen Aktion völlig ahnungslos in die nächste. Als das raubtierkapitalistische China eine ganze Region zugesperrt hat war eigentlich klar wie gefährlich dieses Virus ist. Und das war im Januar. Man sollte den Menschen klar machen dass ihr Leben nie wieder so sein wird wie vorher. Eine Impfung ändert daran gar nichts. Es gibt nur eine Möglichkeit, keiner geht rein und keiner geht raus. China hats gemacht, Australien und Neuseeland ebenfalls, dort ist es verschwunden, ist bei uns nicht möglich, also weiter wie bisher. Kann man so machen, ist aber wesentlich ineffizienter und teurer als ein totaler shutdown für 4 Wochen. ', ' @Zeit Sind es wirklich Positivtests oder Neuinfektionen? Positivtests kann man auch zweimal erhalten (bspw. wird man an Tag 3 positiv getestet und dann macht man einen zweiten Test nach 10 Tagen, ist aber weiterhin positiv), "Neuinfektionen" kann man nur einmal haben. Das RKI übermittelt v.a. die Neuinfektionen und nicht die Positivquote (die man aber der allg. Testquote ablesen kann). ', ' Was für ein Schwachsinn. Deutschland ist definitiv besser durch die Pandemie gekommen als der Schnitt der Welt. Gem. Worldometer ist Deutschland sowohl bei der Zahl der Infizierten und der Toten pro 1.000.000 Einwohner unter dem Durchschnitt der Welt. Viele Länder haben aber deutlich weniger getestet + eine höhere Übersterblichkeit. Ich kann nicht verstehen, wie man so einen (obj. betrachtet) Müll erzählen kann, ohne dass das von der Zeit richtig eingeordnet wird (nämlich, dass falsche Angaben von ihm gemacht wurden). ', ' Also ich lasse mich von dieser ganzen Panikmache nicht anstecken. Das sind doch alles nur Hypothesen. Kein Wissenschaftler kann in die Zukunft schauen. Alle diese Katastrophenszenarien basieren doch nur auf Berechnungen von Computermodellen. Wenn man da einen bestimmten geschätzten Parameter nur um 0.01 ändern, kommt ein komplett anderes Ergebnis raus. Nein, man muss schon ehrlich sein und den menschlichen Fehler mit einberechnen. Der Mensch irrt wo er steht und geht! Und in der Wissenschaft gilt das besonders. Wir können doch nicht Milliarden Euro ausgeben und unsere Wirtschaft schwächen, auf der Basis solcher halbgarer Hypothesen! Wissenschaft ist nun mal keine Demokratie. Die Mehrheit entscheidet nicht. Bei Galileo hat nur einer gesagt: Und sie bewegt sich doch. Und die Mehrheit der Gelehrten bevorzugte dennoch das geozentrische Weltbild. Also wie gesagt: Das Klima ist etwas komplexes und globales, da kann kein einzelnes Land irgendwas ausrichten, noch dazu wenn es sich damit wirtschaftlich zugrunde richtet und wenn gleichzeitig große Länder wie China oder Indien ihre Wirtschaft erst noch so richtig aufblasen und weiter entwickeln. Da müssen wir konkurrenzfähig bleiben. Das beste wäre, wir passen uns an die Klimaveränderungen an. Panikmache und Hysterie bringen gar nichts. Genauso wenig wie Verbote und Einschränkungen. Wir müssen einen kühlen Kopf bewahren und Erfindergeist zeigen. ', ' << Um eine Webseite mit einer DDoS-Attacke zu stören, sind kaum technische Kenntnisse notwendig. >> Das ist der springende Punkt. Das kann das Scriptkiddie in der Nachbarschaft sein aber auch sogenannte Querdenker, die denen da oben mal eins auswischen wollen. Sollte jemand aus der Gruppierung dahinter stecken, sollte man die betreffende Person(en) schnell finden und zur Verantwortung ziehen. ', ' Es gibt zwei Dinge die Unendlich sind - dass Universum und die Dummheit der Menschen. Beim Ersten bin ich mir jedoch nicht sicher. A. EINSTEIN << Am Samstag meldete die britische Statistikbehörde mehr als 23.000 Neuinfektionen. Dennoch demonstrierten in London am Wochenende Tausende für ein Ende der Corona-Beschränkungen. Die Protestierenden sprachen sich gegen die Maskenpflicht aus und kritisierten die geltenden Maßnahmen als Tyrannei oder Überwachung oder stellten die Pandemie an sich in Frage. >> ', ' Also für mich persönlich ändert sich weder mit Kontaktbeschränkungen, noch mit Einschränkungen und Schließungen in Tourismus, Gastronomie, Sport und Kultur etwas, da würde nichts anders laufen als ich es eh gerade schon handhabe. Mir ist bewusst, dass das nicht für Unternehmer, Betreiber und Wirte gilt. Die gehören MASSIV finanziell unterstützt und voll ausgeglichen, ebenso Soloselbständige. Das könnte man mit einer Steuer machen... anfangen könnte man ganz oben mit der Vermögenssteuer. Dass ein paar wenige einzelne Menschen die ganze Krisenzeit von mehreren Millionen Menschen problemlos tragen könnte, ohne essenziell auf lebenswichtige Dinge wie Kaviar oder Dividenden verzichten zu müssen, stößt mir übel auf. Ich bin für einen harten, aber gut terminierten und nicht unendlich langen, sondern absehbaren Lockdown (ein "Ende in Sicht" schont zumindest etwas die Psyche), der dann aber finanziell, sozial, psychisch und solidarisch von allen anderen getragen werden muss, die dazu beitragen können. Schulen ab Mittelstufe sollten auch auf Homeschooling setzen in der gleichen Zeit, um die jüngeren Klassen besser in Präsenz auf die Klassenräume verteilen zu können. Denke bei den Jüngeren würde das schwieriger, von der Entwicklung, dem Verständnis, der Umsetzung her. Dafür könnten die Älteren das abfedern, indem sie zu Hause sind. Es muss alles zusammenwirken, damit man den vollen positiven Effekt auf die Kurve hat. Hoffentlich! ', ' "Bei der langfristigen absoluten Höhe gibt es keinen Unterschied. Da nun aber in erster Linie die Beobachtung des Trends wichtig ist, sollte man sich auf eine Quelle einigen, bzw. die menschlichen Ressourcen, die zur Erhebung der redundanten ZEIT- Zahlen aufgewendet werden einem sinnvollen Zweck zuführen." Ich stimme umfassend zu! Allerdings fiel vor einer Woche dadurch auf, dass das RKI ca. 1100 Fälle aus Niedersachsen und Saarland "vergessen" hatte, die knapp 1000  aus Niedersachsen waren Montags früh schon öffentlich. Korrektur oder Hinweis gab es keine/n. ', ' "Die Einstufung als Risikogebiet erfolgt, wenn ein Land oder eine Region den Grenzwert von 50 Neuinfektionen auf 100.000 Einwohner in den vergangenen sieben Tagen überschreitet." Diese Inzidenzwerte werden sowohl von Änderungen im Testungsumfang als auch durch Unterschiede von Land zu Land beim Testungsumfang und/oder bei der Teststrategie genauso stark beeinflusst wie durch die Änderungen bei der Virusverbreitung. Man arbeitet hier also mit selbstgezimmerten Hausnummern. Schon innerhalb Ds hackt es beim Vergleich zwischen Frühjahr und Herbst gewaltig. Die 50/100.000 aus dem April/Mai entsprechen jetzt etwa 150 bis 200/100.000. Bitte von Prof. Antes erklären lassen! ', ' Covid-19, Wachstumsrate und Prognosen zu „Fallzahlen“ 23.11.2020, Meldung 10.864 aktuelle tägl. Wachstums-, Ausbreitungsrate: -0,5 ± 1,5%, für 16.11.20 3-Tagesprognose zu Fallzahlen**: 12.800, 17.000, 21.000 Das Ausbreitungsgeschehen für Gesamtdeutschland stagniert bei nur leicht negativer Wachstumsrate. Die Rate zeigt starke Schwankungen um 0%, ein Trend ist seit 16.11.2020 nicht erkennbar (gleichbedeutend mit R_t = 1, effektiver R-Wert***). Bericht vom 22.11.2020, Meldung 15.741 aktuelle tägl. Wachstums-, Ausbreitungsrate: -0,5 ± 1,5%, für 15.11.20 3-Tagesprognose zu Fallzahlen**: 11.600, 12.800, 17.000 Das Ausbreitungsgeschehen für Gesamtdeutschland stagniert. Die Wachstumsrate zeigt starke Schwankungen um 0%, ein Trend ist seit 16.11.2020 aktuell nicht erkennbar (gleichbedeutend mit R_t = 1, effektiver R-Wert***). Hinweise Beispiele statistischer Auswertemöglichkeiten von Herrn Lippold: https://covid19-de-stats.sou…. ** Alle Fallzahlen des RKI weisen starke Verzerrungen auf, die für die Nutzung im Rahmen einer epidemiologischen Kurve meist stark überhöhte Zahlenwerte darstellen. *** vergl. hier https://rtlive.de ']
textsam = 'In this study project, we focus on doppelgänger detection in online communities through linguistic details on texts that authors published under their online pseudonyms. In our scenario, a doppelgänger refers to a user maintaining multiple online identities in a given online platform. In addition, we assume that doppelgängers try to remain undiscovered, i.e., side channel information, such as IP addresses or email accounts, is circumvented, e.g., due to the use of anonymization techniques, and, thus, cannot be used in the detection approach. In particular, we rely on the fact that consistently maintaining separate writing styles is hard. Therefore, we aim to answer the question whether detecting doppelgängers in comment sections of news sites is feasible by only leveraging stylometric information. Stylometry, which studies the linguistic style of texts, is a well-suited approach here as it requires no additional information except for text samples to extract author-specific characteristics from. In particular, our detection algorithm will consists of three steps: (1) collection of a sufficient number of news comments posted by various Internet users, (2) extraction of possibly expressive patterns, i.e., features, from these comments, and (3) implementation of an efficient doppelgänger detector for news comments. In the first practical sheet, we gathered a large number of user comments posted in response to articles in a newspaper. In the next part of the study project, we focus on the second step of the doppelgänger detection algorithm, in which we have to retrieve meaningful stylistic features from the collected user comments.'
digit_reg = re.compile("[^0-9]")
# char_reg = re.compile("[^a-zA-ZäöüÄÖÜ]*")

punctuation = string.punctuation


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


def preprocess(tokens, fs=True):
    """Return text length, vocabulary size and optionally the frequency
    spectrum.
    :param fs: additionally calculate and return the frequency
               spectrum
    """
    text_length = len(tokens)
    vocabulary_size = len(set(tokens))
    frequency_list = collections.Counter(tokens)
    frequency_spectrum = dict(collections.Counter(frequency_list.values()))
    return text_length, frequency_spectrum


def simpson_d(text_length, frequency_spectrum):
    return sum((freq_size * (freq / text_length) * ((freq - 1) / (text_length - 1)) for freq, freq_size in frequency_spectrum.items()))


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


def multiple_whitespace(text):
    count = text.count('  ')
    return count


# count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
def count_punctuation(text):
    return len(list(filter(lambda c: c in string.punctuation, text)))

def lang_detect(text):

    result = pd.DataFrame(text)
    pd.options.display.width = 0
    # result["sentiment"] = data["content"].apply(lambda x:TextBlob(x).sentiment.polarity)

    result["lang"] = data["content"].apply(lambda x: langdetect.detect(x) if
    x.strip() != "" else "")
    print(result.head())


def wordfreq_counter(text):
    dd = stop_words(text)
    counter = Counter(dd)
    freq_words = Counter(counter).most_common(20)
    print(counter)
    return freq_words


for row in sample:
    strip_row = row.strip()
    split_by_word = strip_row.split()
    tokenList.append(split_by_word)
    ease_r = textstat.flesch_reading_ease(row)
    ease_reading.append(ease_r)
    fog = textstat.gunning_fog(row)
    gunning_fog.append(fog)



    sentences = sent_tokenize(strip_row)
    sentenceCount += len(sentences)

    puncCountPerComm = count_punctuation(strip_row)
    puncCountPerCommTuple = strip_row, puncCountPerComm
    puncCountPerCommList.append(puncCountPerCommTuple)

    multiSpacePerCommTuple = strip_row, multiple_whitespace(strip_row)
    multiSpacePerCommList.append(multiSpacePerCommTuple)

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

        if wordCountPerSentence >= 20:
            longSenCount += 1
        elif wordCountPerSentence <= 10:
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

    grammarChkPerCommTuple = strip_row, grammarChkCnt
    grammarChkPerCommList.append(grammarChkPerCommTuple)

    sentenceLenPerCommList.append(sentenceLength / len(sentences))

    numberOfWordsPerComm.append(len(split_by_word))

    # tags = nltk.pos_tag(split_by_word)
    blob = TextBlob(strip_row)
    tags = blob.tags
    nounPhrase = blob.noun_phrases
    nounPhraseList.append(nounPhrase)
    typeTokens.append(tags)

    upperCntPerComm = 0

    for word in split_by_word:
        if any(u.isupper() for u in word):
            upperCntPerComm += 1

        charList.append(len(digit_reg.findall(word)))
        charCount += len(digit_reg.findall(word))

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

    upperWordPerCommTuple = row, upperCntPerComm
    upperWordPerCommList.append(upperWordPerCommTuple)

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
    ratio = el, '{0:2.2f}%'.format((100.0 * cnt)/sbase)
    typeTokenRatio.append(ratio)


avg_num_char = charCount/wordCount
avg_num_upper = upperCount/wordCount
avg_num_lower = lowerCount/wordCount
avg_num_digit = digitCount/wordCount

avg_sentence_per_comm = sentenceCount/len(contentList)
avg_num_word_per_sentence = tot_wordCountPerSentence/sentenceCount

a_out = open('task2_a.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 a) Word-level", file=a_out)
print("- Average number of characters per word: ", avg_num_char, file=a_out)
print("- Average number of uppercase letters per word: ", avg_num_upper, file=a_out)
print("- Average number of lowercase letters per word: ", avg_num_lower, file=a_out)
print("- Average number of digits per word: ", avg_num_digit, file=a_out)
print("- Total words per comment: ", numberOfWordsPerComm, file=a_out)
print("- Frequency of large words per comment: ", largeWordCountList, file=a_out)
a_out.close()

b_out = open('task2_b.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 b) Vocabulary-richness", file=b_out)
print("- Number of syllables per word: ", syllablesList, file=b_out)
print("- Type-token Ratio: ", typeTokenRatio, file=b_out)
print("- Entropy of different vocabulary items in a comment: ", entropyList, file=b_out)
print("- Simpson's D measure: ", simpsonList, file=b_out)
print("- Sichel's S measure: ", sichelList, file=b_out)
b_out.close()

c_out = open('task2_c.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 c) Sentence-level", file=c_out)
print("- Number of short sentences: ", shortSenCount, file=c_out)
print("- Number of long sentences: ", longSenCount, file=c_out)
print("- Average sentence length(in characters) per comment: ", sentenceLenPerCommList, file=c_out)
print("- Average sentence length per comment: ", avg_sentence_per_comm, file=c_out)
print("- Flesch-Kincaid grade level of sentences: ", fleschList, file=c_out)
print("- Average number of words per sentence: ", avg_num_word_per_sentence, file=c_out)
c_out.close()

d_out = open('task2_d.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 d) Leetspeak-based", file=d_out)
print("- Original sample text: ", textsam, file=d_out)
print("- Sample text after applying fraction of leetspeak: ", leetspeak(textsam), file=d_out)
d_out.close()

e_out = open('task2_e.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 e)", file=e_out)
print("- Frequency of used punctuation per sentence: ", puncCountPerSenList, file=e_out)
print("- Frequency of used punctuation per comment: ", puncCountPerCommList, file=e_out)
print("- Frequency of repeated occurrence of whitespace per sentence: ", multiSpacePerSenList, file=e_out)
print("- Frequency of repeated occurrence of whitespace per comment", multiSpacePerCommList, file=e_out)
e_out.close()

f_out = open('task2_f.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 f) Content-based", file=f_out)
print("- Average positivity per word: ", totPosPerWord/wordCount, file=f_out)
print("- Average positivity per sentence: ", totPosPerSen/sentenceCount, file=f_out)
print("- Average sensitivity per word: ", totNegPerWord/wordCount, file=f_out)
print("- Average sensitivity per sentence: ", totNegPerSen/sentenceCount, file=f_out)
f_out.close()

g_out = open('task2_g.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 g) Idiosyncratic", file=g_out)
print("- Number of grammar mistakes per sentence: ", grammarChkPerSenList, file=g_out)
print("- Number of grammar mistakes per comment: ", grammarChkPerCommList, file=g_out)
print("- Uppercase word usage per sentence: ", upperWordPerSenList, file=g_out)
print("- Uppercase word usage per comment: ", upperWordPerCommList, file=g_out)
g_out.close()

h_out = open('task2_h.txt', 'w', encoding='utf-8')
print("===============================================================================")
print("task2 h) Additional features", file=h_out)
print("- Noun Phrase: ", nounPhraseList, file=h_out)
print("- Named Entity Recognition: ", list(set(NERList)), file=h_out)
print("- Language Detection :", lang_detect(contentList), file=h_out)
print("- Top 10 words in the content", wordfreq_counter(contentList), file=h_out)
print("- Ease reading for the content", list(set(ease_reading)), file=h_out)
print("- Gunning Fog value for the content", list(set(gunning_fog)), file=h_out)

h_out.close()