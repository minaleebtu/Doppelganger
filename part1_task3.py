import requests
from bs4 import BeautifulSoup
from part1_task2 import userUrls
import mysql.connector
from csv import writer

# DB connect info
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "doppelganger"
)
mycursor = mydb.cursor()

count = 1
for userUrl in userUrls:
    eachUrl = requests.get(userUrl)
    eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")

    username = eachUrlParse.find("h2",{"class":"user-header__title"}).get_text()
    commentNumParse = eachUrlParse.find(class_="user-profile__info").select('span:last-child')

    def trimComment(commentNum):
        index = commentNum.lstrip().find('Kommentar')
        return commentNum.lstrip()[:index].rstrip()

    def getCommentNum(commentNumParse):
        for commentNum in commentNumParse:
            if '.' in commentNum.text:
                return trimComment(commentNum.text.replace('.', ''))
            else:
                return trimComment(commentNum.text)

    def getPageNum(pageNum):
        for page in pageNum:
            if '.' in page.text:
                return page.text.replace('.', '')
            else:
                return page.text

    def getarticleTitle(articleTitle):
        index = articleTitle.lstrip().find(':')
        return articleTitle.lstrip()[index + 1:].lstrip()

    commentNum = int(getCommentNum(commentNumParse))

    if commentNum >= 100:
        count += 1
        pageNum = eachUrlParse.select('section > div > div > div > ul > li:last-child')
        pageNumValue = int(getPageNum(pageNum))
        commentUrls = []

        for x in range(1, pageNumValue + 1):
            commentUrls.append(userUrl + "?p=" + str(x))

        for commentUrl in commentUrls:
            eachCommentUrl = requests.get(commentUrl)
            eachCommentUrlParse = BeautifulSoup(eachCommentUrl.content, "html.parser")
            username = eachCommentUrlParse.find(class_="user-header__title").get_text()
            commentContainer = eachCommentUrlParse.find_all("article", {"class": "user-comment"})

            for comment in commentContainer:
                content = comment.find("div", {"class": "user-comment__text"}).text.replace('\n', ' ')
                articleTitle = comment.find("a", {"class": "user-comment__link"}).get_text().rstrip()
                commDate = comment.find("time", {"class": "user-comment__date"}).get_text()

                if len(content.split()) >= 50:
                    sql = "INSERT IGNORE INTO comments (articleTitle, username, content, commDate) VALUES (%s, %s, %s, %s)"
                    val = (getarticleTitle(articleTitle), username, content, commDate)
                    mycursor.execute(sql, val)
    if count == 50:
        break

mydb.commit()
mycursor.execute("select * from comments")
res = mycursor.fetchall()
selectTotalNum = mycursor.rowcount
mydb.close()

with open('comments_incl_per_user.csv', 'w', encoding='utf8') as csv_file:
    csv_writer = writer(csv_file)
    headers = ['Article Title', 'Username', 'Content of Comment', 'Date']
    csv_writer.writerow(headers)
    csv_writer.writerows(res)
