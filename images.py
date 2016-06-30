from posixpath import basename,dirname
from urlparse import urlsplit
from urlparse import urlparse
from bs4 import BeautifulSoup
from PIL import Image
import multiprocessing
import urllib2
import json
import os
import re

def process_url(raw_url):
 if ' ' not in raw_url[-1]:
     raw_url=raw_url.replace(' ','%20')
     return raw_url
 elif ' ' in raw_url[-1]:
     raw_url=raw_url[:-1]
     raw_url=raw_url.replace(' ','%20')
     return raw_url

# get the images first and then join them later... required if parallelized later
def getImg(url):
    parse_object=urlparse(url)
    dirname=basename(parse_object.path)

    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]

    urlcontent=opener.open(url).read()
    soup = BeautifulSoup(urlcontent, "lxml")
    images = soup.findAll("img")
    imgurls=re.findall('img .*src="(.*?)"',urlcontent)

    dirname=basename(parse_object.path)
    if not os.path.exists('images'):
        os.mkdir("images")
    try:
        os.mkdir("images/"+dirname)
    except:
        pass
    os.chdir("images/"+dirname)
    collected_images = []
    for image in images:
        try:
            imgurl=re.findall('img .*src="(.*?)"', str(image))[0]
            if imgurl[-3:] != "svg":
                imgurl=process_url(imgurl)

                if 'height' in str(image):
                    if image['height'] > 80 and image['width'] > 80:
                        collected_images.append(image)
                        print (imgurl, image["alt"], image['height'], image['width'])

                else:
                    imgdata=urllib2.urlopen(imgurl).read()
                    if len(imgdata) > 1000:
                        collected_images.append(image)
                        print image

                # filname=basename(urlsplit(imgurl)[2])
                # output=open(filname,'wb')
                # output.write(imgdata)
                # output.close()
                # os.remove(filename)
        except:
            pass


if __name__ == '__main__':
    url = raw_input("enter website to get images from\n")
    images = getImg(url)