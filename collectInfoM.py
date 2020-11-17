import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from csv import writer

url = "https://www.zeit.de/"

pagetoparse = requests.get(url)

Soup = BeautifulSoup(pagetoparse.content, "html.parser")

dataList = []

titles = Soup.find_all(class_='zon-teaser-standard__title')
author = Soup.find_all(attrs={'class' : 'zon-teaser-standard__byline'})
comment = Soup.find_all(attrs={'class' : 'zon-teaser-standard__commentcount js-link-commentcount js-update-commentcount'})
urls = Soup.select('div> div > article> a')

leadArticles = Soup.find_all(class_='zon-teaser-lead')
articles = Soup.find_all(class_='zon-teaser-standard')

for leadArticle in leadArticles:
    if leadArticle.find(class_='zon-teaser-lead__commentcount js-link-commentcount js-update-commentcount'):
        title = leadArticle.find(class_='zon-teaser-lead__title').get_text()
        summary = leadArticle.find(class_='zon-teaser-lead__text').get_text()
        url = leadArticle.find(class_='zon-teaser-lead__combined-link')['href']
        # author = leadArticle.find(class_='zon-teaser-lead__byline').get_text()
        # commentNum = leadArticle.find(class_='zon-teaser-lead__commentcount js-link-commentcount js-update-commentcount').get_text()

        eachUrl = requests.get(url)
        eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")
        if eachUrlParse.select('div > div.byline > span > a > span'):
            author = eachUrlParse.select('div > div > span > a > span')
        elif eachUrlParse.find(class_='metadata__source'):
            author = eachUrlParse.select('div > div > span')
        else:
            author = 'none'

        if eachUrlParse.find(class_='metadata__commentcount js-scroll'):
            commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
        elif eachUrlParse.find(class_='comment-section__headline'):
            commentNum = eachUrlParse.find(class_='comment-section__headline').get_text()
        else:
            commentNum = 'none'
        # commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()

        if eachUrlParse.find(class_='metadata__date'):
            pubDate = eachUrlParse.find(class_='metadata__date')
        else:
            pubDate = eachUrlParse.find(class_='meta__date encoded-date')


        dataTuple = title, summary, url, author, commentNum, pubDate
        print(author, url, commentNum, pubDate)
        dataList.append(dataTuple)

for article in articles:
    if article.find(class_='zon-teaser-standard__commentcount js-link-commentcount js-update-commentcount'):
        title = article.find(class_='zon-teaser-standard__title').get_text()
        summary = article.find(class_='zon-teaser-standard__text').get_text()
        url = article.find(class_='zon-teaser-standard__combined-link')['href']
        # author = article.find(class_='zon-teaser-standard__byline').get_text()
        # commentNum = article.find(class_='zon-teaser-standard__commentcount js-link-commentcount js-update-commentcount').get_text()

        eachUrl = requests.get(url)
        eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")

        if eachUrlParse.select('div > div.byline > span > a > span'):
            author = eachUrlParse.select('div > div > span > a > span')
        elif eachUrlParse.find(class_='metadata__source'):
            author = eachUrlParse.select('div > div > span')
        else:
            author = 'none'

        if eachUrlParse.find(class_='metadata__commentcount js-scroll'):
            commentNum = eachUrlParse.find(class_='metadata__commentcount js-scroll').get_text()
        elif eachUrlParse.find(class_='comment-section__headline'):
            commentNum = eachUrlParse.find(class_='comment-section__headline').get_text()
        else:
            commentNum = 'none'

        if eachUrlParse.find(class_='metadata__date'):
            pubDate = eachUrlParse.find(class_='metadata__date')
        else:
            pubDate = eachUrlParse.find(class_='meta__date encoded-date')
        # pubDate = eachUrlParse.find(class_='metadata__date')

        dataTuple = title, summary, url, author, commentNum, pubDate
        print(author, url, commentNum, pubDate)
        dataList.append(dataTuple)
# print(dataList)

