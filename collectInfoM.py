import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from csv import writer
import pandas as pd

url = "https://www.zeit.de/"

pagetoparse = requests.get(url)

Soup = BeautifulSoup(pagetoparse.content, "html.parser")

dataList = []
articles = Soup.find_all(class_=['main main--centerpage','zon-teaser-standard', 'zon-teaser-classic', 'zon-teaser-lead', 'zon-teaser-wide', 'zon-teaser-standard_metadata', 'zon-teaser-lead_commentcount js-link-commentcount js-update-commentcount'])

for article in articles:
    # try:
        title = article.find(class_=['zon-teaser-lead__title', 'zon-teaser-standard__title', 'zon-teaser-classic__title', 'zon-teaser-wide__title']).get_text()
        # date = article.find(class_=['metadata_date' 'meta_date encoded-date'])
        url = article.find(class_=['zon-teaser-lead__combined-link', 'zon-teaser-standard__combined-link', 'zon-teaser-classic__combined-link', 'zon-teaser-wide__combined-link'])['href']
        # comments = article.select('js-update-commentcount')

        eachUrl = requests.get(url)
        eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")

        if eachUrlParse.select('div > div > span > a > span'):
            author = eachUrlParse.select('div > div > span > a > span')
        elif eachUrlParse.find(class_='metadata__source'):
            author = eachUrlParse.find(class_='metadata__source')
        else:
            author = 'none'

        def hasNumbers(inputString):
            return any(char.isdigit() for char in inputString)
        def trimComment(commentNum):
            index = commentNum.lstrip().find('Kommentar')
            return commentNum.lstrip()[:index]

        commentNumParse = eachUrlParse.find(class_=['metadata__commentcount js-scroll', 'comment-section__headline'])
        if commentNumParse:
            commentNum = commentNumParse.get_text()
            if hasNumbers(commentNum):
                commentNum = trimComment(commentNum)
            else:
                commentNum = 'none'
        else:
            commentNum = 'none'

        if eachUrlParse.find('time'):
            date = eachUrlParse.find('time').get_text()
        else:
            date = 'none'

        # date = article.find(class_=['metadata_date' 'meta_date encoded-date'])

        print(title, "/", url, "/", date, "/", commentNum, "/", author)


    # except Exception as e:
    #     title = ''
    #     print(e)


# leadArticles = Soup.find_all(class_='zon-teaser-lead')
# articles = Soup.find_all(class_='zon-teaser-standard')
# wideArticles = Soup.find_all(class_='zon-teaser-wide')

# for leadArticle in leadArticles:
#
#     if leadArticle.find(class_='zon-teaser-standard'):
#         title = leadArticle.find(class_='zon-teaser-standard__title').get_text()
#     else:
#         title = leadArticle.find(class_='zon-teaser-lead__title').get_text()
#     # title = leadArticle.find(class_='zon-teaser-lead__title').get_text()
#     summary = leadArticle.find(class_='zon-teaser-lead__text').get_text()
#     url = leadArticle.find(class_='zon-teaser-lead__combined-link')['href']
#     # author = leadArticle.find(class_='zon-teaser-lead__byline').get_text()
#     # commentNum = leadArticle.find(class_='zon-teaser-lead__commentcount js-link-commentcount js-update-commentcount').get_text()
#
#     eachUrl = requests.get(url)
#     eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")
#     if eachUrlParse.select('div > div > span > a > span'):
#         authorPre = eachUrlParse.select('div > div > span > a > span')
#         # print(type(authorPre))
#         for x in authorPre:
#             author = x.get_text()
#     elif eachUrlParse.find(class_='metadata__source'):
#         authorPre = eachUrlParse.select('div > div > span')
#         for x in authorPre:
#             author = x.get_text()
#     else:
#         author = 'none'
#
#     if eachUrlParse.find(class_='metadata__commentcount js-scroll'):
#         commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
#     elif eachUrlParse.find(class_='comment-section__headline'):
#         if eachUrlParse.find(class_='comment-section__headline').get_text().isdigit():
#             commentNum = eachUrlParse.find(class_='comment-section__headline').get_text()
#         else:
#             commentNum = 'none'
#     else:
#         commentNum = 'none'
#     # commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
#
#     if eachUrlParse.find('time'):
#         pubDate = eachUrlParse.find('time').get_text()
#     else:
#         pubDate = 'none'
#     # if eachUrlParse.find(class_='metadata__date'):
#     #     pubDate = eachUrlParse.find(class_='metadata__date').get_text()
#     # elif eachUrlParse.find(class_='meta__date encoded-date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date encoded-date').get_text()
#     # elif eachUrlParse.find(class_='meta__date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date').get_text()
#     # else:
#     #     pubDate = 'none'
#
#     dataTuple = title, summary, url, author, commentNum, pubDate
#     print("lead: ", title, url, pubDate)
#     # print('commentNum: ', dataTuple[dataTuple.index(commentNum)])
#     if dataTuple[dataTuple.index(commentNum)]!= 'none':
#         dataList.append(dataTuple)
#
# for leadArticle in leadArticles:
#
#     if leadArticle.find(class_='zon-teaser-standard'):
#         title = leadArticle.find(class_='zon-teaser-standard__title').get_text()
#     else:
#         title = leadArticle.find(class_='zon-teaser-lead__title').get_text()
#     # title = leadArticle.find(class_='zon-teaser-lead__title').get_text()
#     summary = leadArticle.find(class_='zon-teaser-lead__text').get_text()
#     url = leadArticle.find(class_='zon-teaser-lead__combined-link')['href']
#     # author = leadArticle.find(class_='zon-teaser-lead__byline').get_text()
#     # commentNum = leadArticle.find(class_='zon-teaser-lead__commentcount js-link-commentcount js-update-commentcount').get_text()
#
#     eachUrl = requests.get(url)
#     eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")
#     if eachUrlParse.select('div > div > span > a > span'):
#         authorPre = eachUrlParse.select('div > div > span > a > span')
#         # print(type(authorPre))
#         for x in authorPre:
#             author = x.get_text()
#     elif eachUrlParse.find(class_='metadata__source'):
#         authorPre = eachUrlParse.select('div > div > span')
#         for x in authorPre:
#             author = x.get_text()
#     else:
#         author = 'none'
#
#     if eachUrlParse.find(class_='metadata__commentcount js-scroll'):
#         commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
#     elif eachUrlParse.find(class_='comment-section__headline'):
#         if eachUrlParse.find(class_='comment-section__headline').get_text().isdigit():
#             commentNum = eachUrlParse.find(class_='comment-section__headline').get_text()
#         else:
#             commentNum = 'none'
#     else:
#         commentNum = 'none'
#     # commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
#
#     if eachUrlParse.find('time'):
#         pubDate = eachUrlParse.find('time').get_text()
#     else:
#         pubDate = 'none'
#     # if eachUrlParse.find(class_='metadata__date'):
#     #     pubDate = eachUrlParse.find(class_='metadata__date').get_text()
#     # elif eachUrlParse.find(class_='meta__date encoded-date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date encoded-date').get_text()
#     # elif eachUrlParse.find(class_='meta__date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date').get_text()
#     # else:
#     #     pubDate = 'none'
#
#     dataTuple = title, summary, url, author, commentNum, pubDate
#     print("lead: ", title, url, pubDate)
#     # print('commentNum: ', dataTuple[dataTuple.index(commentNum)])
#     if dataTuple[dataTuple.index(commentNum)]!= 'none':
#         dataList.append(dataTuple)
#
# for wideArticle in wideArticles:
#
#     title = wideArticle.find(class_='zon-teaser-wide__title').get_text()
#     summary = wideArticle.find(class_='zon-teaser-wide__text').get_text()
#     url = wideArticle.find(class_='zon-teaser-wide__combined-link')['href']
#
#     eachUrl = requests.get(url)
#     eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")
#     if eachUrlParse.select('div > div > span > a > span'):
#         authorPre = eachUrlParse.select('div > div > span > a > span')
#         # print(type(authorPre))
#         for x in authorPre:
#             author = x.get_text()
#     elif eachUrlParse.find(class_='metadata__source'):
#         authorPre = eachUrlParse.select('div > div > span')
#         for x in authorPre:
#             author = x.get_text()
#     else:
#         author = 'none'
#
#     if eachUrlParse.find(class_='metadata__commentcount js-scroll'):
#         commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
#     elif eachUrlParse.find(class_='comment-section__headline'):
#         if eachUrlParse.find(class_='comment-section__headline').get_text().isdigit():
#             commentNum = eachUrlParse.find(class_='comment-section__headline').get_text()
#         else:
#             commentNum = 'none'
#     else:
#         commentNum = 'none'
#
#     if eachUrlParse.find('time'):
#         pubDate = eachUrlParse.find('time').get_text()
#     else:
#         pubDate = 'none'
#     # if eachUrlParse.find(class_='metadata__date'):
#     #     pubDate = eachUrlParse.find(class_='metadata__date').get_text()
#     # elif eachUrlParse.find(class_='meta__date encoded-date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date encoded-date').get_text()
#     # elif eachUrlParse.find(class_='meta__date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date').get_text()
#     # else:
#     #     pubDate = 'none'
#
#     dataTuple = title, summary, url, author, commentNum, pubDate
#     print("wide: ", title, url, pubDate)
#     if dataTuple[dataTuple.index(commentNum)]!= 'none':
#         dataList.append(dataTuple)
#
# for article in articles:
#     title = article.find(class_='zon-teaser-standard__title').get_text()
#     summary = article.find(class_='zon-teaser-standard__text').get_text()
#     url = article.find(class_='zon-teaser-standard__combined-link')['href']
#     # author = article.find(class_='zon-teaser-standard__byline').get_text()
#     # commentNum = article.find(class_='zon-teaser-standard__commentcount js-link-commentcount js-update-commentcount').get_text()
#
#     eachUrl = requests.get(url)
#     eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")
#
#     if eachUrlParse.select('div > div > span > a > span'):
#         authorPre = eachUrlParse.select('div > div > span > a > span')
#         # print(type(authorPre))
#         for x in authorPre:
#             author = x.get_text()
#     elif eachUrlParse.find(class_='metadata__source'):
#         authorPre = eachUrlParse.select('div > div > span')
#         for x in authorPre:
#             author = x.get_text()
#     else:
#         author = 'none'
#
#     if eachUrlParse.find(class_='metadata__commentcount js-scroll'):
#         commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
#     elif eachUrlParse.find(class_='comment-section__headline'):
#         if eachUrlParse.find(class_='comment-section__headline').get_text().isdigit():
#             commentNum = eachUrlParse.find(class_='comment-section__headline').get_text()
#         else:
#             commentNum = 'none'
#         # commentNum = eachUrlParse.find(class_='comment-section__headline').get_text()
#     else:
#         commentNum = 'none'
#
#     if eachUrlParse.find('time'):
#         pubDate = eachUrlParse.find('time').get_text()
#     else:
#         pubDate = 'none'
#     # if eachUrlParse.find(class_='metadata__date'):
#     #     pubDate = eachUrlParse.find(class_='metadata__date').get_text()
#     # elif eachUrlParse.find(class_='meta__date encoded-date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date encoded-date').get_text()
#     # elif eachUrlParse.find(class_='meta__date'):
#     #     pubDate = eachUrlParse.find(class_='meta__date').get_text()
#     # else:
#     #     pubDate = 'none'
#
#     dataTuple = title, summary, url, author, commentNum, pubDate
#     print("standard: ", title, url, pubDate)
#     # print('commentNum: ', dataTuple[dataTuple.index(commentNum)])
#     if dataTuple[dataTuple.index(commentNum)] != 'none':
#         dataList.append(dataTuple)
#
# print(dataList)

