# import collectInfo
import requests
from bs4 import BeautifulSoup
import mysql.connector

mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "root",
    database = "doppelganger"
)

mycursor = mydb.cursor()

# for url in collectInfo.urls:
#     sql = "INSERT INTO articles (title, url, author) VALUES (%s, %s, %s)"
#     val = (url.get('href'),)
#     mycursor.execute(sql, val)
#     print(type(val))

# sql = "select * from articles"
# myresult = mycursor.fetchall()

# for x in myresult:
#   print(x)
mydb.commit()

print(mycursor.rowcount, "data inserted.")