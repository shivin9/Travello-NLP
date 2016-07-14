from sklearn import preprocessing
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
    page_title = soup.select("title")[0].get_text().encode(
        'ascii', 'ignore').strip().encode('ascii')
    paragraphs = []

    for p in raw.split('\n'):
        p = p.strip()
        if len(p) > 2:
            p = p.replace('\t', '')
            p = p.replace('\n', '')
            paragraphs.append(p)

    paragraphs.append(page_title)
    # lens can be computed on it's own
    paradict = {}
    for i in range(len(paragraphs)):
        if paragraphs[i] not in paradict:
            paradict[paragraphs[i]] = i

    # special entry for the title of the page
    paradict[page_title] = -1

    return soup, paragraphs, paradict


def consolidateStuff(url, titles, addresses, images):
    soup, paragraphs, paradict = parsePage(url)
    lens = [len(p) for p in paragraphs]
    jsonoutput = {}
    posspara = LongParas(lens)

    if 'tripadvisor' in url:
        jsonoutput[0] = {'Place Name': paragraphs[0],
                      'Write-up': paragraphs[posspara[0]],
                      'Address': addresses
                      }

    else:
        addrs = []
        # the head of addresses
        for address in addresses:
            addrs.append(paradict[address[0]])

        addrs = np.array(addrs)
        fullThing = getFull(titles, addrs, posspara)
        # print fullThing

        for i in range(len(fullThing)):
            onething = fullThing[i]
            jsonoutput[i] = {'Place Name': paragraphs[onething[0]],
                             'Write-up': paragraphs[posspara[onething[1]]],
                             'Address': addresses[onething[2]]
                             }

    choices = [str(image) for image in images]

    for i in range(len(titles)):
        rightImage = process.extractOne(jsonoutput[i]['Place Name'],
                                        choices)[0]
        imgurls = re.findall('img .*src="(.*?)"', rightImage)
        jsonoutput[i]['Image URL'] = imgurls[0]

    # print jsonoutput
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
    # print "len(posspara) = " + str(len(posspara))
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


# get the images first and then join them later... required if
# parallelized later
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
            data1[i] = np.array(getvec([paras[i]])[:NUM_FEATURES])

        data = np.zeros((BATCH_SIZE * (batches), SEQ_LENGTH, NUM_FEATURES))
        for i in range(len(data1)):
            data[i / SEQ_LENGTH, i % SEQ_LENGTH, :] = data1[i]
        del(data1)

    else:
        batches = len1 / BATCH_SIZE + 1
        data = np.zeros((batches * BATCH_SIZE, NUM_FEATURES))
        for i in range(len1):
            data[i / BATCH_SIZE, :] = np.array(getvec([paras[i]])[:NUM_FEATURES])

    return data


def getScores(pred, paras, params):
    X = getData(paras, params['NUM_FEATURES'], params[
                'BATCH_SIZE'], SEQ_LENGTH=params['SEQ_LENGTH'])

    res = pred(X).flatten()
    out = []
    for i in range(len(paras)):
        out.append((paras[i], res[i]))
    return out


def load_dataset(X, y, wndw=1):
    '''
        wndw is the window_size for buffering the input with 0 vectors
    '''
    for i in range(len(X)):
        X[i] = np.array(X[i])

    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')

    X_train = X[:-1000]
    y_train = y[:-1000]

    X_val = X[-1000:]
    y_val = y[-1000:]

    if wndw / 2 > 0:
        num_feat = len(X[0])
        Xbuffer = np.zeros((wndw / 2, num_feat))
        ybuffer = np.zeros((wndw / 2,))
        X_train = np.vstack([Xbuffer, X_train, Xbuffer])
        X_val = np.vstack([Xbuffer, X_val, Xbuffer])

        # append 0s at the front and the back of both training and testing labels
        y_train = np.append(ybuffer, y_train)
        y_train = np.append(y_train, ybuffer)

        y_val = np.append(ybuffer, y_val)
        y_val = np.append(y_val, ybuffer)

    return X_train, y_train, X_val, y_val