from address import parsepage
from title import getTitle
from images import getImg
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import json
import re

def getStuff(url):
    addresses = parsepage(url)
    fullThing = getTitle(url, addresses)
    images = getImg(url)

    choices = [str(image) for image in images]

    for i in range(len(fullThing)):
        rightImage = process.extractOne(fullThing[i]['Place Name'], choices)[0]
        imgurls=re.findall('img .*src="(.*?)"', rightImage)
        fullThing[i]['Image URL'] = imgurls[0]

    fullThing = json.dumps(fullThing, indent=4)
    print fullThing
    fullThing = fullThing.replace('\n', '<br>')
    return fullThing