import requests
from bs4 import BeautifulSoup
import part1_task1
from csv import writer
import mysql.connector

commentTuple = ()
commentList = []
for articleData in part1_task1.articleList:
    title = articleData[0]
    articleUrl = articleData[1]

    eachUrl = requests.get(articleUrl)
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
            return str(date)[index:].replace('\n','')

    pageNumValue = getPageNum(pageNum)
    # print("pageNum", pageNumValue)
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
            # print(commentUrl)

            eachCommentUrl = requests.get(commentUrl)
            eachCommentUrlParse = BeautifulSoup(eachCommentUrl.content,"html.parser")

            # comments = eachCommentUrlParse.find_all(class_='comment__container')
            comments = eachCommentUrlParse.find_all("article",{"class":"comment"})

            for comment in comments:
                # print("comment:",comment)
                username = comment.find(class_="comment-meta__name")
                content = comment.find(class_='comment__body').text.replace('\n','')
                commDate = comment.find("a",{"data-ct-label": "datum"})
                userUrl = username['href']


                if username != None:
                    username = username.get_text().strip()
                else:
                    username = 'none'

                if userUrl == None:
                    userUrl = 'none'

                commentTuple = title, username, content, trimDate(commDate), userUrl
                print(commentTuple)
                # print("title: ", title)
                # print("username:",username)
                # print("content:", content)
                # print("commDate: ", trimDate(commDate))
                if len(commentTuple[commentTuple.index(content)].split()) >= 50:
                    commentList.append(commentTuple)
                    print("more than 50(",len(commentTuple[commentTuple.index(content)].split()),")")
                else:
                    print("nopeeeeeeeee")

    else:
        comments = eachUrlParse.find_all("article", {"class": "comment"})
        for comment in comments:
            username = comment.find(class_="comment-meta__name")
            content = comment.find(class_='comment__body').text.replace('\n','')
            commDate = comment.find("a", {"data-ct-label": "datum"})
            userUrl = username['href']

            if username != None:
                username = username.get_text().strip()
            else:
                username = 'none'

            if userUrl == None:
                userUrl = 'none'

            commentTuple = title, username, content, trimDate(commDate), userUrl
            # print("title: ", title)
            # print("username: ", username)
            # print("content: ", content)
            # print("commDate: ", trimDate(commDate))
            print(commentTuple)
            if len(commentTuple[commentTuple.index(content)].split()) >= 50:
                commentList.append(commentTuple)
                print("more than 50(",len(commentTuple[commentTuple.index(content)].split()),")")
            else:
                print("nopeeeeeeeee")
    # print(commentTuple)
    # contentLen = len(content.split())
    # print("content: ", content)
    # print("len: ", len(commentTuple[commentTuple.index(content)].split()))

print(commentList)