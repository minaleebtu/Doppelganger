import requests
import urllib
from bs4 import BeautifulSoup

url = "https://www.zeit.de/"

pagetoparse = requests.get (url)

Soup = BeautifulSoup(pagetoparse.content, "html.parser")

header = pagetoparse.headers
titles = Soup.find_all(attrs={'class':'zon-teaser-standard__title'})
author = Soup.find_all(attrs={'class' : 'zon-teaser-standard__byline'})
comment = Soup.find_all(attrs={'class' : 'zon-teaser-standard__commentcount js-link-commentcount js-update-commentcount'})
urls = Soup.select('div> div > article> a')
#print('title: ', titles)



for title in titles:
    print(' Title : \n', title.text)

for auth in author:
    print('Author :\n', ' '.join(auth.text.split()))

for comm in comment:
    print(comm.text)
# urls = Soup.find_all(attrs={'class':'zon-teaser-standard__combined-link'})

for url in urls:
    print(url.get('href'))

#authors = Soup.select('div > section > article > a > div > div > span > span > font > font')
#print('authors: ', authors)

