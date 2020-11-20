import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from csv import writer
import pandas as pd
import mysql.connector

# DB connect info
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "doppelganger"
)

url = "https://www.zeit.de/"

pagetoparse = requests.get(url)

Soup = BeautifulSoup(pagetoparse.content, "html.parser")

articleTuple = ()
articleList = []

articles = Soup.find_all(class_=['main main--centerpage','zon-teaser-standard', 'zon-teaser-classic', 'zon-teaser-lead', 'zon-teaser-wide', 'zon-teaser-standard_metadata', 'zon-teaser-lead_commentcount js-link-commentcount js-update-commentcount'])

# get article info
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

    # check acomment number data has number or not
    def hasNumbers(commentNum):
        return any(char.isdigit() for char in commentNum)
    # trim comment number to exact number
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

    articleTuple = title, url, author, commentNum, date

    # put data to article List if comment number of article is not null
    if articleTuple[articleTuple.index(commentNum)] != 'none':
        articleList.append(articleTuple)

    # print(dataTuple)

# get comment info by using article data from article List
commentTuple = ()
commentList = []
for articleData in articleList:

    # print(articleData)
    title = articleData[0]
    url = articleData[1]

    eachUrl = requests.get(url)
    eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")

    pageNum = eachUrlParse.select('div > div > span > small')
    # get number of pages of comments
    def getPageNum(pageNum):
        for x in pageNum:
            index = x.get_text().find('von')
            return x.get_text()[index+4:]
    def trimDate(commDate):
        for date in commDate:
            index = str(date).find('vor')
            return str(date)[index:].lstrip()

    pageNumValue = getPageNum(pageNum)
    print("pageNum", pageNumValue)
    # only for those which have page number of comments
    if pageNumValue != None:
        pageNumValueInt = int(pageNumValue)
        # print(pageNumValueInt)
        commentUrls = []
        for x in range(1,pageNumValueInt+1):
            commentUrls.append(articleData[1] + "?page=" + str(x) + "#comments")

        # print(commentUrls)

        # parse each comment page per article
        for commentUrl in commentUrls:
            print(commentUrl)

            eachCommentUrl = requests.get(commentUrl)
            eachCommentUrlParse = BeautifulSoup(eachCommentUrl.content,"html.parser")

            # comments = eachCommentUrlParse.find_all(class_='comment__container')
            comments = eachCommentUrlParse.find_all("article",{"class":"comment"})

            for comment in comments:
                print("title: ", title)
                # print("comment:",comment)
                username = comment.find(class_="comment-meta__name")
                content = comment.find(class_='comment__body').text
                commDate = comment.find("a",{"data-ct-label": "datum"})

                if username != None:
                    username = username.get_text().strip()
                else:
                    username = 'none'

                print("username:",username)
                print("content:", content)
                print("commDate: ", trimDate(commDate))

    else:
        comments = eachUrlParse.find_all("article", {"class": "comment"})
        for comment in comments:
            print("title: ", title)
            username = comment.find(class_="comment-meta__name")
            content = comment.find(class_='comment__body').text
            commDate = comment.find("a", {"data-ct-label": "datum"})

            if username != None:
                username = username.get_text().strip()
            else:
                username = 'none'
            print("username: ", username)
            print("content: ", content)
            print("commDate: ", trimDate(commDate))

    # except Exception as e:
    #     title = ''
    #     print(e)
