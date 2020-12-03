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

# sys.stdout = open('result.txt', 'w', encoding='utf-8')

textstat.set_lang("de")

nltk.download('averaged_perceptron_tagger')

tool = language_tool_python.LanguageTool('de')

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

sample = [' Die europäischen Regierungen stolpern von einer sinnlosen Aktion völlig ahnungslos in die nächste. Als das raubtierkapitalistische China eine ganze Region zugesperrt hat war eigentlich klar wie gefährlich dieses Virus ist. Und das war im Januar. Man sollte den Menschen klar machen dass ihr Leben nie wieder so sein wird wie vorher. Eine Impfung ändert daran gar nichts. Es gibt nur eine Möglichkeit, keiner geht rein und keiner geht raus. China hats gemacht, Australien und Neuseeland ebenfalls, dort ist es verschwunden, ist bei uns nicht möglich, also weiter wie bisher. Kann man so machen, ist aber wesentlich ineffizienter und teurer als ein totaler shutdown für 4 Wochen. ', ' @Zeit Sind es wirklich Positivtests oder Neuinfektionen? Positivtests kann man auch zweimal erhalten (bspw. wird man an Tag 3 positiv getestet und dann macht man einen zweiten Test nach 10 Tagen, ist aber weiterhin positiv), "Neuinfektionen" kann man nur einmal haben. Das RKI übermittelt v.a. die Neuinfektionen und nicht die Positivquote (die man aber der allg. Testquote ablesen kann). ', ' Was für ein Schwachsinn. Deutschland ist definitiv besser durch die Pandemie gekommen als der Schnitt der Welt. Gem. Worldometer ist Deutschland sowohl bei der Zahl der Infizierten und der Toten pro 1.000.000 Einwohner unter dem Durchschnitt der Welt. Viele Länder haben aber deutlich weniger getestet + eine höhere Übersterblichkeit. Ich kann nicht verstehen, wie man so einen (obj. betrachtet) Müll erzählen kann, ohne dass das von der Zeit richtig eingeordnet wird (nämlich, dass falsche Angaben von ihm gemacht wurden). ', ' Also ich lasse mich von dieser ganzen Panikmache nicht anstecken. Das sind doch alles nur Hypothesen. Kein Wissenschaftler kann in die Zukunft schauen. Alle diese Katastrophenszenarien basieren doch nur auf Berechnungen von Computermodellen. Wenn man da einen bestimmten geschätzten Parameter nur um 0.01 ändern, kommt ein komplett anderes Ergebnis raus. Nein, man muss schon ehrlich sein und den menschlichen Fehler mit einberechnen. Der Mensch irrt wo er steht und geht! Und in der Wissenschaft gilt das besonders. Wir können doch nicht Milliarden Euro ausgeben und unsere Wirtschaft schwächen, auf der Basis solcher halbgarer Hypothesen! Wissenschaft ist nun mal keine Demokratie. Die Mehrheit entscheidet nicht. Bei Galileo hat nur einer gesagt: Und sie bewegt sich doch. Und die Mehrheit der Gelehrten bevorzugte dennoch das geozentrische Weltbild. Also wie gesagt: Das Klima ist etwas komplexes und globales, da kann kein einzelnes Land irgendwas ausrichten, noch dazu wenn es sich damit wirtschaftlich zugrunde richtet und wenn gleichzeitig große Länder wie China oder Indien ihre Wirtschaft erst noch so richtig aufblasen und weiter entwickeln. Da müssen wir konkurrenzfähig bleiben. Das beste wäre, wir passen uns an die Klimaveränderungen an. Panikmache und Hysterie bringen gar nichts. Genauso wenig wie Verbote und Einschränkungen. Wir müssen einen kühlen Kopf bewahren und Erfindergeist zeigen. ', ' << Um eine Webseite mit einer DDoS-Attacke zu stören, sind kaum technische Kenntnisse notwendig. >> Das ist der springende Punkt. Das kann das Scriptkiddie in der Nachbarschaft sein aber auch sogenannte Querdenker, die denen da oben mal eins auswischen wollen. Sollte jemand aus der Gruppierung dahinter stecken, sollte man die betreffende Person(en) schnell finden und zur Verantwortung ziehen. ', ' Es gibt zwei Dinge die Unendlich sind - dass Universum und die Dummheit der Menschen. Beim Ersten bin ich mir jedoch nicht sicher. A. EINSTEIN << Am Samstag meldete die britische Statistikbehörde mehr als 23.000 Neuinfektionen. Dennoch demonstrierten in London am Wochenende Tausende für ein Ende der Corona-Beschränkungen. Die Protestierenden sprachen sich gegen die Maskenpflicht aus und kritisierten die geltenden Maßnahmen als Tyrannei oder Überwachung oder stellten die Pandemie an sich in Frage. >> ', ' Also für mich persönlich ändert sich weder mit Kontaktbeschränkungen, noch mit Einschränkungen und Schließungen in Tourismus, Gastronomie, Sport und Kultur etwas, da würde nichts anders laufen als ich es eh gerade schon handhabe. Mir ist bewusst, dass das nicht für Unternehmer, Betreiber und Wirte gilt. Die gehören MASSIV finanziell unterstützt und voll ausgeglichen, ebenso Soloselbständige. Das könnte man mit einer Steuer machen... anfangen könnte man ganz oben mit der Vermögenssteuer. Dass ein paar wenige einzelne Menschen die ganze Krisenzeit von mehreren Millionen Menschen problemlos tragen könnte, ohne essenziell auf lebenswichtige Dinge wie Kaviar oder Dividenden verzichten zu müssen, stößt mir übel auf. Ich bin für einen harten, aber gut terminierten und nicht unendlich langen, sondern absehbaren Lockdown (ein "Ende in Sicht" schont zumindest etwas die Psyche), der dann aber finanziell, sozial, psychisch und solidarisch von allen anderen getragen werden muss, die dazu beitragen können. Schulen ab Mittelstufe sollten auch auf Homeschooling setzen in der gleichen Zeit, um die jüngeren Klassen besser in Präsenz auf die Klassenräume verteilen zu können. Denke bei den Jüngeren würde das schwieriger, von der Entwicklung, dem Verständnis, der Umsetzung her. Dafür könnten die Älteren das abfedern, indem sie zu Hause sind. Es muss alles zusammenwirken, damit man den vollen positiven Effekt auf die Kurve hat. Hoffentlich! ', ' "Bei der langfristigen absoluten Höhe gibt es keinen Unterschied. Da nun aber in erster Linie die Beobachtung des Trends wichtig ist, sollte man sich auf eine Quelle einigen, bzw. die menschlichen Ressourcen, die zur Erhebung der redundanten ZEIT- Zahlen aufgewendet werden einem sinnvollen Zweck zuführen." Ich stimme umfassend zu! Allerdings fiel vor einer Woche dadurch auf, dass das RKI ca. 1100 Fälle aus Niedersachsen und Saarland "vergessen" hatte, die knapp 1000  aus Niedersachsen waren Montags früh schon öffentlich. Korrektur oder Hinweis gab es keine/n. ', ' "Die Einstufung als Risikogebiet erfolgt, wenn ein Land oder eine Region den Grenzwert von 50 Neuinfektionen auf 100.000 Einwohner in den vergangenen sieben Tagen überschreitet." Diese Inzidenzwerte werden sowohl von Änderungen im Testungsumfang als auch durch Unterschiede von Land zu Land beim Testungsumfang und/oder bei der Teststrategie genauso stark beeinflusst wie durch die Änderungen bei der Virusverbreitung. Man arbeitet hier also mit selbstgezimmerten Hausnummern. Schon innerhalb Ds hackt es beim Vergleich zwischen Frühjahr und Herbst gewaltig. Die 50/100.000 aus dem April/Mai entsprechen jetzt etwa 150 bis 200/100.000. Bitte von Prof. Antes erklären lassen! ', ' Covid-19, Wachstumsrate und Prognosen zu „Fallzahlen“ 23.11.2020, Meldung 10.864 aktuelle tägl. Wachstums-, Ausbreitungsrate: -0,5 ± 1,5%, für 16.11.20 3-Tagesprognose zu Fallzahlen**: 12.800, 17.000, 21.000 Das Ausbreitungsgeschehen für Gesamtdeutschland stagniert bei nur leicht negativer Wachstumsrate. Die Rate zeigt starke Schwankungen um 0%, ein Trend ist seit 16.11.2020 nicht erkennbar (gleichbedeutend mit R_t = 1, effektiver R-Wert***). Bericht vom 22.11.2020, Meldung 15.741 aktuelle tägl. Wachstums-, Ausbreitungsrate: -0,5 ± 1,5%, für 15.11.20 3-Tagesprognose zu Fallzahlen**: 11.600, 12.800, 17.000 Das Ausbreitungsgeschehen für Gesamtdeutschland stagniert. Die Wachstumsrate zeigt starke Schwankungen um 0%, ein Trend ist seit 16.11.2020 aktuell nicht erkennbar (gleichbedeutend mit R_t = 1, effektiver R-Wert***). Hinweise Beispiele statistischer Auswertemöglichkeiten von Herrn Lippold: https://covid19-de-stats.sou…. ** Alle Fallzahlen des RKI weisen starke Verzerrungen auf, die für die Nutzung im Rahmen einer epidemiologischen Kurve meist stark überhöhte Zahlenwerte darstellen. *** vergl. hier https://rtlive.de ']
textsam = 'In this study project, we focus on doppelgänger detection in online communities through linguistic details on texts that authors published under their online pseudonyms. In our scenario, a doppelgänger refers to a user maintaining multiple online identities in a given online platform. In addition, we assume that doppelgängers try to remain undiscovered, i.e., side channel information, such as IP addresses or email accounts, is circumvented, e.g., due to the use of anonymization techniques, and, thus, cannot be used in the detection approach. In particular, we rely on the fact that consistently maintaining separate writing styles is hard. Therefore, we aim to answer the question whether detecting doppelgängers in comment sections of news sites is feasible by only leveraging stylometric information. Stylometry, which studies the linguistic style of texts, is a well-suited approach here as it requires no additional information except for text samples to extract author-specific characteristics from. In particular, our detection algorithm will consists of three steps: (1) collection of a sufficient number of news comments posted by various Internet users, (2) extraction of possibly expressive patterns, i.e., features, from these comments, and (3) implementation of an efficient doppelgänger detector for news comments. In the first practical sheet, we gathered a large number of user comments posted in response to articles in a newspaper. In the next part of the study project, we focus on the second step of the doppelgänger detection algorithm, in which we have to retrieve meaningful stylistic features from the collected user comments.'
digit_reg = re.compile("[^0-9]")
char_reg = re.compile("[^a-zA-ZäöüÄÖÜ]*")


def shannon_entropy(words):
    length = len(words)
    freqs = Counter(words)
    distribution = np.array(list(freqs.values()))
    distribution = np.true_divide(distribution, length)

    E = stats.entropy(distribution, base=2)

    return E

def simpson_d(num_tokens, freq_spectrum):
    """Calculate Simpson’s D.
    Parameters:
        num_tokens (int): Absolute number of tokens.
        freq_spectrum (pd.Series): Counted occurring frequencies.
    """
    a = freq_spectrum.values / num_tokens
    b = freq_spectrum.index.values - 1
    return (freq_spectrum.values * a * (b / (num_tokens - 1))).sum()

def sichel_s(num_types, freq_spectrum):
    """Calculate Sichel’s S (1975).
    Used formula:
        .. math::
            S = \frac{D}{V}
    Parameters:
        num_types (int): Absolute number of types.
        freq_spectrum (dict): Counted occurring frequencies.
    """
    return freq_spectrum[2] / num_types

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

# def paragraphs(self):
#     if self._paragraphs is not None:
#         for p in  self._paragraphs:
#             yield p
#     else:
#         raw_paras = self.raw_text.split(self.paragraph_delimiter)
#         gen = (Paragraph(self, p) for p in raw_paras if p)
#         self._paragraphs = []
#         for p in gen:
#             self._paragraphs.append(p)
#             yield p


for row in sample:
    strip_row = row.strip()
    split_by_word = strip_row.split()
    tokenList.append(split_by_word)

    sentences = sent_tokenize(strip_row)
    sentenceCount += len(sentences)

    sentenceLength = 0

    grammarChkCnt = 0

    largeWordCount = 0

    for sentence in sentences:
        wordsInSen = sentence.split()
        upperCntPerSen = 0

        for eachWord in wordsInSen:
            if any(u.isupper() for u in eachWord):
                upperCntPerSen += 1

        upperWordPerSenTuple = sentence, upperCntPerSen
        upperWordPerSenList.append(upperWordPerSenTuple)

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

    grammarChkPerCommTuple = strip_row, grammarChkCnt
    grammarChkPerCommList.append(grammarChkPerCommTuple)

    sentenceLenPerCommList.append(sentenceLength / len(sentences))

    numberOfWordsPerComm.append(len(split_by_word))

    tags = nltk.pos_tag(split_by_word)
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

# print("===============================================================================")
# print("task2 a) Word-level")
# print("- Average number of characters per word: ", avg_num_char)
# print("- Average number of uppercase letters per word: ", avg_num_upper)
# print("- Average number of lowercase letters per word: ", avg_num_lower)
# print("- Average number of digits per word: ", avg_num_digit)
# print("- Total words per comment: ", numberOfWordsPerComm)
# print("- Frequency of large words per comment: ", largeWordCountList)
#
# print("===============================================================================")
# print("task2 b) Vocabulary-richness")
# print("- Number of syllables per word: ", syllablesList)
# print("- Type-token Ratio: ", typeTokenRatio)
# print("- Entropy of different vocabulary items in a comment: ", entropyList)
# print("- Simpson's D measure: ", "XXXXXXXXX")
# print("- Sichel's S measure: ", "XXXXXXXXX")
#
# print("===============================================================================")
# print("task2 c) Sentence-level")
# print("- Number of short sentences: ", shortSenCount)
# print("- Number of long sentences: ", longSenCount)
# print("- Average sentence length(in characters) per comment: ", sentenceLenPerCommList)
# print("- Average sentence length per comment: ", avg_sentence_per_comm)
# print("- Flesch-Kincaid grade level of sentences: ", fleschList)
# print("- Average number of words per sentence: ", avg_num_word_per_sentence)
# print("===============================================================================")
# print("task2 d) Leetspeak-based")
# print("- Original sample text: ", textsam)
# print("- Sample text after applying fraction of leetspeak: ", leetspeak(textsam))

# print("===============================================================================")
# print("task2 g) Idiosyncratic")
# print("- Number of grammar mistakes per sentence: ", grammarChkPerSenList)
# print("- Number of grammar mistakes per comment: ", grammarChkPerCommList)
print("- Uppercase word usage per sentence: ", upperWordPerSenList)
print("- Uppercase word usage per comment: ", upperWordPerCommList)
