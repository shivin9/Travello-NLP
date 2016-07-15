from sklearn import preprocessing
from fuzzywuzzy import process
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import numpy as np
import urllib2
import json
import re
from create_training import getvec


def parsePage(url):
    '''
    This method is very important and it used at many places in the package. It is used to
    parse a page and extract relevant information from the page ie. the text and it creates
    two other data structures which are helpful for other computations all throughout

    Parameters
    ----------
    url : The url of the page

    Returns
    -------
    soup : It is a BeautifulSoup object which has parsed the webpage are separated it based
        on the html tags. For more information please refer to the BeautifulSoup library

    paragraphs : It is a list of all the paragraphs on the webpage. Paragraphs are pieces
        of text which are separated by '\n'. Throughout the documentation we have used the
        term index of a paragraph which is nothing but it's index in this list

    paradict : It is a python dictionary which stores the reverse of paragraphs ie. the
        indices are referred to by the paragraphs which they index to
    '''

    opener = urllib2.build_opener()
    # this header is important as many websites detect that the request is coming from
    # a python bot and they reject the request. This header is to make the request look
    # as if it's coming from an authentic browser
    opener.addheaders = [
        ('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')

    # remove the styling and scripting tags
    for elem in soup.findAll(['script', 'style']):
        elem.extract()

    raw = soup.get_text().encode('ascii', 'ignore')

    # page_title is the title of the page which is not present in raw
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
    paradict = {}
    for i in range(len(paragraphs)):
        if paragraphs[i] not in paradict:
            paradict[paragraphs[i]] = i

    # special entry for the page_title at the end of the list
    paradict[page_title] = -1

    return soup, paragraphs, paradict


def consolidateStuff(url, titles, addresses, images):
    '''
    This method is used to 'glue' the correct titles, addresses, write-ups
    and the images to form a blob of complete information about a particular point
    of interest in the webpage

    Parameters
    ----------
    url : The url of the page

    titles : The data structure as returned by the getTitles method

    addresses : The addresses as returned by the getAddress function. Note that
        these addresses are already consolidated amongst themselves

    images : The url of images as returned by the getImage method.

    Returns
    -------
    jsonoutput : A stringified dictionary where indices(starting from 0) repsesent another
        dictionary indexed by 'Place Name', 'Write-up', 'Address' and 'Image URL'

    '''
    soup, paragraphs, paradict = parsePage(url)
    lens = [len(p) for p in paragraphs]
    jsonoutput = {}
    posspara = LongParas(lens)
    titles = [paradict[t] for t in titles]

    # special consilidation for TripAdvisor
    if 'tripadvisor' in url:
        jsonoutput[0] = {'Place Name': paragraphs[-1],
                      'Write-up': paragraphs[posspara[0]],
                      'Address': addresses
                      }

    else:
        addrs = []

        # the head of addresses as explained in the getTitles method
        for address in addresses:
            addrs.append(paradict[address[0]])

        addrs = np.array(addrs)

        # to glue the titles, addresses and the write-ups to form a correct
        # and complete blob of information
        fullThing = getFull(titles, addrs, posspara)

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
    '''
    This method returns the long write-ups in the webpage by clustering them into 2
    clusters based on their lengths. Idea is that kmeans will separate the longer
    paragraphs

    Parameters
    ----------
    lens : A list where each element represents the length of that corresponding
        paragraphs at that index

    Returns
    -------
    posspara : A list of indices of the long paragraphs
    '''

    res = np.array(lens)
    # reshapre the res array to form a matrix of size (1, len(res))
    res = res.reshape(-1, 1)
    est = KMeans(n_clusters=2)
    est.fit(res)
    labels = est.labels_
    # since the cluster numbers are randomly assigned, we need to find the cluster number
    # the long paragraphs by finding out in which cluster the longest paragraph lies
    bestpara = np.argmax(res)
    reqlabel = labels[bestpara]

    # these are the possible paragraphs about the restaurants
    posspara = np.where(labels == reqlabel)[0]
    # print "len(posspara) = " + str(len(posspara))
    return posspara


def findmin(arr):
    '''
    This is a simple method to find the least non-negative number in an array. Could'nt
    find an easier way to do this task and therefore had to write it in a new method

    Parameters
    ----------
    arr : A numpy array

    Returns
    -------
    The smallest non-negative number in arr

    '''
    maxx = np.max(arr)
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = maxx + 1
    return np.argmin(arr)


def getFull(headers, addresses, possparas):
    '''
    A very small but important method which for every header(take note) and not the
    address finds the appropriate address object and the write-up by using loaclity
    arguments described elsewhere. Now we can appreciate the passing of header and address
    information by their indices rather than the text itself

    Parameters
    ----------
    headers : A list of indices of all the header paragraphs

    addresses : A list of indices of all the first line of addresses... only first
        line is needed as we will make locality arguments

    possparas : A list of indices of all the write-ups

    Returns
    -------
    blob : A list of shape=(number_of_pts_of_interests, 3) (Hopefully!)
        Each element of blog is itself a list which has the indices of the header,
        it's corresponding write-up and address
    '''
    out = []
    for header in headers:
        parapos = findmin(possparas - header)
        addrpos = findmin(addresses - header)
        out.append([header, parapos, addrpos])
    blob = np.array(out)
    return blob


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