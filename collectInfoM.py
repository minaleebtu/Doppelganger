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

for article in articles:
    # try:
    title = article.find(class_=['zon-teaser-lead__title', 'zon-teaser-standard__title', 'zon-teaser-classic__title', 'zon-teaser-wide__title']).get_text()
    # date = article.find(class_=['metadata_date' 'meta_date encoded-date'])
    url = article.find(class_=['zon-teaser-lead__combined-link', 'zon-teaser-standard__combined-link', 'zon-teaser-classic__combined-link', 'zon-teaser-wide__combined-link'])['href']
    # comments = article.select('js-update-commentcount')
    authors = []

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
        return commentNum.lstrip()[:index].rstrip()

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

    if dataTuple[dataTuple.index(commentNum)] != 'none':
        dataList.append(dataTuple)

    # urlIndex = dataTuple.index(url)
    # commentNumIndex = dataTuple.index(commentNum)
    # print("commentNumIndex:",commentNumIndex)

    # print(dataTuple)
for data in dataList:

    # print(data)

    eachUrl = requests.get(data[1])
    eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")

    pageNum = eachUrlParse.select('div > div > span > small')
    # print(pageNum)

    def getPageNum(pageNum):
        for x in pageNum:
            index = x.get_text().find('von')
            return x.get_text()[index+4:]

    pageNumValue = getPageNum(pageNum)
    # print(pageNumValue)

    if pageNumValue != None:
        pageNumValueInt = int(pageNumValue)
        # print(pageNumValueInt)
        commentUrls = []
        for x in range(1,pageNumValueInt+1):
            commentUrls.append(data[1] + "?page=" + str(x) + "#comments")

        # print(commentUrls)

        for commentUrl in commentUrls:
            print(commentUrl)
            eachCommentUrl = requests.get(commentUrl)
            eachCommentUrlParse = BeautifulSoup(eachCommentUrl.content,"html.parser")

            # comments = eachCommentUrlParse.find_all(class_='comment__container')
            comments = eachCommentUrlParse.find_all("article",{"class":"comment"})

            for comment in comments:
                # print("comment:",comment)
                userId = comment.find("a",{"data-ct-label": "user_profile"})
                # userId = comment.select('div > div > h4 > a')
                userIds = []
                for x in userId:
                    userIds.append(str(x))
                    userId = userIds
                print("userIds:",userIds)
                # contentOfComment = comment.find(class_='comment__body')
                # print("content:", contentOfComment)

    # except Exception as e:
    #     title = ''
    #     print(e)
