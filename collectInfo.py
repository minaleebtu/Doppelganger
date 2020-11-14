import requests
import urllib
from bs4 import BeautifulSoup

url = "https://www.zeit.de/"

pagetoparse = requests.get (url)

Soup = BeautifulSoup(pagetoparse.content, "html.parser")

header = pagetoparse.headers
titles = Soup.find_all(attrs={'class':'zon-teaser-standard__title'})
print('title: ', titles)

for title in titles:
    print(title)
# urls = Soup.find_all(attrs={'class':'zon-teaser-standard__combined-link'})
urls = Soup.select('div> div > article> a')

for url in urls:
    print(url.get('href'))

authors = Soup.select('div > section > article > a > div > div > span > span > font > font')
print('authors: ', authors)
