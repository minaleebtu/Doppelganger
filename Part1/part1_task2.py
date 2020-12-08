import requests
from bs4 import BeautifulSoup
from part1_task1 import articleList
import mysql.connector

# DB connect info
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "doppelganger"
)
mycursor = mydb.cursor()

commentTuple = ()
commentList = []
userUrlList = []

for articleData in articleList:
    articleTitle = articleData[0]
    articleUrl = articleData[1]

    eachUrl = requests.get(articleUrl)
    eachUrlParse = BeautifulSoup(eachUrl.content, "html.parser")

    pageNum = eachUrlParse.select('div > div > span > small')

    # get number of pages of comments
    def getPageNum(pageNum):
        for x in pageNum:
            index = x.get_text().find('von')
            if '.' in x.get_text()[index+4:]:
                return x.get_text()[index + 4:].replace('.', '')
            else:
                return x.get_text()[index+4:]
    def trimDate(commDate):
        for date in commDate:
            index = str(date).find('vor')
            return str(date)[index:].replace('\n', '').rstrip()

    pageNumValue = getPageNum(pageNum)

    # only for those which have page number of comments
    if pageNumValue != None:
        pageNumValueInt = int(pageNumValue)
        commentUrls = []
        for x in range(1, pageNumValueInt+1):
            commentUrls.append(articleData[1] + "?page=" + str(x) + "#comments")

        # parse each comment page per article
        for commentUrl in commentUrls:
            eachCommentUrl = requests.get(commentUrl)
            eachCommentUrlParse = BeautifulSoup(eachCommentUrl.content, "html.parser")

            comments = eachCommentUrlParse.find_all("article", {"class": "comment"})

            for comment in comments:
                username = comment.find(class_="comment-meta__name")
                content = comment.find(class_='comment__body').text.replace('\n', ' ')
                commDate = comment.find("a", {"data-ct-label": "datum"})

                if username != None:
                    username = username.get_text().strip()
                else:
                    username = 'none'

                if comment.find("a",{"data-ct-label":"user_profile"}):
                    userUrl = comment.find("a",{"data-ct-label":"user_profile"})['href']
                else:
                    userUrl = 'none'

                if len(content.split()) >= 50:
                    commentTuple = articleTitle, username, content, trimDate(commDate)
                    commentList.append(commentTuple)
                    userUrlList.append(userUrl)
                else:
                    continue

    else:
        comments = eachUrlParse.find_all("article", {"class": "comment"})
        for comment in comments:
            username = comment.find(class_="comment-meta__name")
            content = comment.find(class_='comment__body').text.replace('\n', ' ')
            commDate = comment.find("a", {"data-ct-label": "datum"})

            if username != None:
                username = username.get_text().strip()
            else:
                username = 'none'

            if comment.find("a", {"data-ct-label": "user_profile"}):
                userUrl = comment.find("a", {"data-ct-label": "user_profile"})['href']
            else:
                userUrl = 'none'

            if len(content.split()) >= 50:
                commentTuple = articleTitle, username, content, trimDate(commDate)
                commentList.append(commentTuple)

                userUrlList.append(userUrl)
            else:
                continue

for articleTitle, username, content, commDate in commentList:
    sql = "INSERT IGNORE INTO comments (articleTitle, username, content, commDate) VALUES (%s, %s, %s, %s)"
    val = (articleTitle, username, content, commDate)
    mycursor.execute(sql, val)

mydb.commit()
mydb.close()

userUrls = list(set(userUrlList))

articleTitles = []
usernames = []
commentsTotalLength = int(0)

for commentData in commentList:
    articleTitles.append(commentData[0])
    usernames.append(commentData[1])
    commentsTotalLength += len(commentData[2])

articleTitleCount = set(articleTitles)
usernamesCount = set(usernames)
commentsTotalNum = len(commentList)

print("Total number of collected articles: ", len(articleTitleCount))
print("Total number of collected users: ", len(usernamesCount))
print("Total number of collected user comments: ", commentsTotalNum)

print("Average number of comments per user: ", commentsTotalNum/len(usernamesCount))
print("Average of comment length: ", commentsTotalLength/commentsTotalNum)