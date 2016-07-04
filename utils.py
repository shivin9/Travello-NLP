from fuzzywuzzy import process
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import numpy as np
import urllib2
import json
import re
from create_training import getvec


# will parse the page only once
def parsePage(url):
    opener = urllib2.build_opener()
    opener.addheaders = [
        ('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')

    for elem in soup.findAll(['script', 'style']):
        elem.extract()

    raw = soup.get_text().encode('ascii', 'ignore')
    paragraphs = []

    for p in raw.split('\n'):
        p = p.strip()
        if len(p) > 2:
            p = p.replace('\t', '')
            p = p.replace('\n', '')
            paragraphs.append(p)

    # lens can be computed on it's own
    paradict = {}
    for i in range(len(paragraphs)):
        if paragraphs[i] not in paradict:
            paradict[paragraphs[i]] = i

    images = getImg(url)
    return soup, paragraphs, images, paradict


def consolidateStuff(url, titles, addresses, images):
    soup, paragraphs, _, paradict = parsePage(url)
    lens = [len(p) for p in paragraphs]

    addrs = []
    # the head of addresses
    for address in addresses[0]:
        addrs.append(paradict[address[0]])
    addrs = np.array(addrs)

    posspara = LongParas(lens)
    fullThing = getFull(titles, addrs, posspara)
    jsonoutput = {}

    for i in range(len(fullThing)):
        onething = fullThing[i]
        jsonoutput[i] = {'Place Name': paragraphs[onething[0]],
                         'Write-up': paragraphs[posspara[onething[1]]],
                         'Address': addresses[0][onething[2]]
                         }

    choices = [str(image) for image in images]

    for i in range(len(fullThing)):
        rightImage = process.extractOne(jsonoutput[i]['Place Name'], choices)
        imgurls = re.findall('img .*src="(.*?)"', rightImage[0])
        jsonoutput[i]['Image URL'] = imgurls[0]

    return json.dumps(jsonoutput, indent=4)


def LongParas(lens):
    res = np.array(lens)
    res = res.reshape(-1, 1)
    est = KMeans(n_clusters=2)
    est.fit(res)
    labels = est.labels_
    bestpara = np.argmax(res)
    reqlabel = labels[bestpara]

    # these are the possible paragraphs about the restaurants
    posspara = np.where(labels == reqlabel)[0]
    return posspara


def findmin(arr):
    maxx = np.max(arr)
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = maxx + 1
    return np.argmin(arr)


def getFull(headers, addresses, possparas):
    out = []
    for header in headers:
        parapos = findmin(possparas - header)
        addrpos = findmin(addresses - header)
        out.append([header, parapos, addrpos])
    return np.array(out)


def process_url(raw_url):
    if ' ' not in raw_url[-1]:
        raw_url = raw_url.replace(' ', '%20')

    elif ' ' in raw_url[-1]:
        raw_url = raw_url[:-1]
        raw_url = raw_url.replace(' ', '%20')

    return raw_url


# get the images first and then join them later... required if parallelized later
def getImg(url):
    opener = urllib2.build_opener()
    opener.addheaders = [
        ('User-agent',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102\ Chrome/50.0.2661.102 Safari/537.36')]

    urlcontent = opener.open(url).read()
    soup = BeautifulSoup(urlcontent, "lxml")
    images = soup.findAll("img")

    collected_images = []

    for image in images:
        try:
            imgurl = re.findall('img .*src="(.*?)"', str(image))[0]
            if imgurl[-3:] != "svg":
                imgurl = process_url(imgurl)

                if 'height' in str(image) and 'width' in str(image):
                    if int(image['height']) > 80 and int(image['width']) > 80:
                        collected_images.append(image)
                        # print (imgurl, image["alt"], image['height'], image['width'])

                else:
                    imgdata = urllib2.urlopen(imgurl).read()
                    if len(imgdata) > 5000:
                        collected_images.append(image)
                        # print (image, len(imgdata))

        except:
            pass
    return collected_images


# Can return the data in required shape
def getData(paras, NUM_FEATURES, BATCH_SIZE, SEQ_LENGTH=None):
    len1 = len(paras)
    if SEQ_LENGTH:
        batches = len1 / (BATCH_SIZE * SEQ_LENGTH) + 1
        data1 = np.zeros((BATCH_SIZE * (batches) * SEQ_LENGTH, NUM_FEATURES))
        for i in range(len1):
            data1[i] = np.array(getvec([paras[i]]))

        data = np.zeros((BATCH_SIZE * (batches), SEQ_LENGTH, NUM_FEATURES))
        for i in range(len(data1)):
            data[i / SEQ_LENGTH, i % SEQ_LENGTH, :] = data1[i]
        del(data1)

    else:
        batches = len1 / BATCH_SIZE + 1
        data = np.zeros((batches * BATCH_SIZE, NUM_FEATURES))
        for i in range(len1):
            data[i / BATCH_SIZE, :] = np.array(getvec([paras[i]]))

    return data
