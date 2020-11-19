import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from csv import writer
import pandas as pd

url = "https://www.zeit.de/"

pagetoparse = requests.get(url)

Soup = BeautifulSoup(pagetoparse.content, "html.parser")

dataTuple = ()
dataList = []

articles = Soup.find_all(class_=['main main--centerpage','zon-teaser-standard', 'zon-teaser-classic', 'zon-teaser-lead', 'zon-teaser-wide', 'zon-teaser-standard_metadata', 'zon-teaser-lead_commentcount js-link-commentcount js-update-commentcount'])
comments = Soup.find_all(class_='comment-section')

for article in articles:
    # try:
    title = article.find(class_=['zon-teaser-lead__title', 'zon-teaser-standard__title', 'zon-teaser-classic__title', 'zon-teaser-wide__title']).get_text()
    # date = article.find(class_=['metadata_date' 'meta_date encoded-date'])
    url = article.find(class_=['zon-teaser-lead__combined-link', 'zon-teaser-standard__combined-link', 'zon-teaser-classic__combined-link', 'zon-teaser-wide__combined-link'])['href']
    # comments = article.select('js-update-commentcount')
    authors = []
    userIds = []

    eachUrl = requests.get(url)
    eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")

    if eachUrlParse.select('div > div > span > a > span'):
        author = eachUrlParse.select('div > div > span > a > span')
        for authorName in author:
            authors.append(authorName.get_text().strip())
            author = authors

    elif eachUrlParse.find(class_='metadata__source'):
        author = 'none'
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

    dataTuple = title, url, author, commentNum, date
    print(dataTuple)
    if dataTuple[dataTuple.index(commentNum)] != 'none':
        dataList.append(dataTuple)

print(dataList)
# pageNums = []
    # if commentNum > 0:
    #     pageNum = eachUrlParse.select('div > div > span > small')
    # print(pageNum[0].text)
    # def getPageNum(pageNum):
    #     for x in pageNum:
    #         pageNums.append(x.get_text())
    #         pageNum = pageNums
    #     return pageNum
    #
    # getPageNum(pageNum)

    # userId = eachUrlParse.select('div > div > h4 > a')
    #
    # for comment in comments:
    #     print("asdfasdfadfasdf")
    #
    #     print(type(userId))
    #     # getPageNum()
    #     for x in userId:
    #         userIds.append(x.get_text().strip())
    #         userId = userIds

    # except Exception as e:
    #     title = ''
    #     print(e)
