import requests
from bs4 import BeautifulSoup
from csv import writer
import mysql.connector

url = "https://www.zeit.de/"

pagetoparse = requests.get(url)

Soup = BeautifulSoup(pagetoparse.content, "html.parser")

articleTuple = ()
articleList = []
titles = []

articles = Soup.find_all(class_=['zon-teaser-standard', 'zon-teaser-classic', 'zon-teaser-lead', 'zon-teaser-wide'])

# get article info
for article in articles:
    title = article.find(class_=['zon-teaser-lead__title', 'zon-teaser-standard__title', 'zon-teaser-classic__title', 'zon-teaser-wide__title']).get_text()
    articleUrl = article.find(class_=['zon-teaser-lead__combined-link', 'zon-teaser-standard__combined-link', 'zon-teaser-classic__combined-link', 'zon-teaser-wide__combined-link'])['href']
    authors = []

    eachUrl = requests.get(articleUrl)
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
        if '.' in commentNum.lstrip()[:index].rstrip():
            return commentNum.lstrip()[:index].rstrip().replace('.', '')
        else:
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
        pubDate = eachUrlParse.find('time').get_text()
    else:
        pubDate = 'none'
    articleTuple = title, articleUrl, author, commentNum, pubDate

    # put data to article List if comment number of article is not null
    if commentNum != 'none':
        articleList.append(articleTuple)

for title in articleList:
    titles.append(title[0])

titleCount = set(titles)

# DB connect info
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "doppelganger"
)

mycursor = mydb.cursor()

for title, articleUrl, author, commentNum, pubDate in articleList:
    sql = "INSERT IGNORE INTO articles (title, articleUrl, author, commentNum, pubDate) VALUES (%s, %s, %s, %s, %s)"
    if author != 'none':
        author = ','.join(author)
    val = (title, articleUrl, author, commentNum, pubDate)
    # print("val: ", val)
    mycursor.execute(sql, val)
mydb.commit()
mycursor.execute("select * from articles")
res = mycursor.fetchall()
mydb.close()

with open('articles.csv', 'w', encoding='utf8') as csv_file:
    csv_writer = writer(csv_file)
    headers = ['Title', 'Title Url', 'Author', 'Comment Number', 'Published Date']

    csv_writer.writerow(headers)
    csv_writer.writerows(res)

print(articleList)