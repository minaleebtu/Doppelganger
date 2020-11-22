import requests
from bs4 import BeautifulSoup
from part1_task1 import articleList
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
        # print(pageNumValueInt)
        commentUrls = []
        for x in range(1, pageNumValueInt+1):
            commentUrls.append(articleData[1] + "?page=" + str(x) + "#comments")

        # parse each comment page per article
        for commentUrl in commentUrls:
            eachCommentUrl = requests.get(commentUrl)
            eachCommentUrlParse = BeautifulSoup(eachCommentUrl.content, "html.parser")

            # comments = eachCommentUrlParse.find_all(class_='comment__container')
            comments = eachCommentUrlParse.find_all("article", {"class": "comment"})

            for comment in comments:
                username = comment.find(class_="comment-meta__name")
                content = comment.find(class_='comment__body').text.replace('\n', ' ')
                commDate = comment.find("a", {"data-ct-label": "datum"})
                # userUrl = comment.find("a",{"data-ct-label":"user_profile"})['href']

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

                    sql = "INSERT IGNORE INTO comments (articleTitle, username, content, commDate) VALUES (%s, %s, %s, %s)"
                    val = (articleTitle, username, content, trimDate(commDate))
                    mycursor.execute(sql, val)
                else:
                    continue

    else:
        comments = eachUrlParse.find_all("article", {"class": "comment"})
        for comment in comments:
            username = comment.find(class_="comment-meta__name")
            content = comment.find(class_='comment__body').text.replace('\n', ' ')
            commDate = comment.find("a", {"data-ct-label": "datum"})
            # userUrl = comment.find("a", {"data-ct-label": "user_profile"})['href']

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

                sql = "INSERT IGNORE INTO comments (articleTitle, username, content, commDate) VALUES (%s, %s, %s, %s)"
                val = (articleTitle, username, content, trimDate(commDate))
                mycursor.execute(sql, val)
            else:
                continue

print(commentList)
mydb.commit()

mycursor.execute("select * from comments")
res = mycursor.fetchall()
selectTotalNum = mycursor.rowcount
mydb.close()

with open('comments.csv', 'w', encoding='utf8') as csv_file:
    csv_writer = writer(csv_file)
    headers = ['Article Title', 'Username', 'Content of Comment', 'Date']
    csv_writer.writerow(headers)
    csv_writer.writerows(res)

userUrls = list(set(userUrlList))
print("userUrls: ", userUrls)
print("userUrlsCount: ", len(userUrls))

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

print("articleTitle(set): ", articleTitleCount)
print("articleTitleCount: ", len(articleTitleCount))

print("usernames: ", usernamesCount)
print("usernamesCount: ", len(usernamesCount))

print("CommentsTotalNum: ", commentsTotalNum)
print("comment total num (from db): ", selectTotalNum)

print("average of comments per user: ", commentsTotalNum/len(usernamesCount))
print("average of comment length: ", commentsTotalLength/commentsTotalNum)
