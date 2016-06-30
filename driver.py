from address import parsepage
from title import getTitle

def getStuff(url):
    addresses = parsepage(url)
    fullThing = getTitle(url, addresses)
    print fullThing
    fullThing = fullThing.replace('\n', '<br>')
    # print fullThing
    return fullThing