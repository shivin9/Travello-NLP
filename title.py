from utils import parsePage, LongParas
import numpy as np
import string
import re

NUM_CLUSTERS = 2


def getTitle(url, addresses=[[1, 2, 3, 4]]):
    soup, paragraphs, paradict = parsePage(url)
    lens = [len(p) for p in paragraphs]
    headerIndices = []
    print len(addresses)

    # separate method for ladyironchef.com
    if 'ladyironchef' in url:
        headerIndices = LICTitle(soup, paradict)

    # trip advisor has only a single place_name/page
    elif 'tripadvisor' in url:
        headerIndices = TripAdTitle(soup, paradict)

    elif len(addresses) <= 3:
        # onetitle = getoneheader(soup, possibleHeaders, paragraphs)
        headerIndices = [-1]

    else:
        # in a few rare cases the title of page can be under <Hn> inside the page also
        # header_titles = [soup.findAll('h'+str(i)) for i in range(1, 5)]
        # header_titles.append(soup.findAll('strong'))
        possibleHeaders = GenPage(soup, paradict)
        # this implies that most probably page is a multi-place blog
        # get the long write-ups about the headings
        posspara = LongParas(lens)

        # generate indices for addresses, they are the first line of address
        addrs = []

        # the head of addresses
        for address in addresses:
            addrs.append(paradict[address[0]])
        addrs = np.array(addrs)

        features = getHeadFeatures(possibleHeaders, addrs, posspara)
        reqindices = np.where(features > 0)[0]

        '''
        # classify the headers
        est = KMeans(n_clusters = NUM_CLUSTERS)
        est.fit(features)
        labels = est.labels_

        print features
        print labels

        # deciding which labels are of real headers
        distarr = []

        for i in range(NUM_CLUSTERS):
            distarr.append(len(np.where(labels==i)[0]))

        distarr = np.array(distarr)
        s = len(addrs)
        distarr = (distarr-s)**2

        reqlabel = np.argmin(distarr)

        print reqlabel
        reqindices = np.where(labels==reqlabel)[0]
        print reqindices'''

        for idx in reqindices:
            headerIndices.append(possibleHeaders[idx])

    print "printing titles"
    for idx in headerIndices:
        print paragraphs[idx]

    return headerIndices


def LICTitle(soup, paradict):
    tags = soup.findAll('span', {"style": "font-size: x-large;"})
    titles = []

    for tag in tags:
        name = tag.get_text().encode('ascii', 'ignore')
        titles.append(name)

    if len(titles) == 0:
        text = soup.findAll(re.compile('strong'))
        for title in text:
            str1 = title.get_text().encode('ascii', 'ignore')
            str1 = str1.replace('\t', '')
            str1 = str1.replace('\n', '')
        if len(str1) > 2:
            titles.append(str1)

    return [paradict[t] for t in titles]


def TripAdTitle(soup, paradict):
    page_title = soup.findAll("title")[0].get_text().encode('ascii', 'ignore')
    for i in range(len(page_title)):
        if page_title[i] in string.punctuation:
            break

    page_title = page_title[0:i]
    return [paradict[page_title]]


def GenPage(soup, paradict):
    headings = soup.findAll(re.compile('h[0-5]|strong'))
    # to remove duplicates, add them to a set
    possheaders = set()
    # to select the elements which have the maximum number of common tags-- failed idea
    # strip the string of waste space from the sides

    for title in headings:
        head = title.get_text().encode('ascii', 'ignore')
        head = head.replace('\t', '')
        head = head.replace('\n', '')
        head = head.strip()
        if len(head) > 2 and (not onlyNumbers(head)) and head in paradict:
            possheaders.add(paradict[head])

    possheaders = sorted(list(possheaders))
    # print "possheaders = " + str(possheaders)
    return possheaders


def getHeadFeatures(headers, addresses, possparas):
    '''
        headers: the indices of the possible headers on the page
        addresses: the indices of the first line of address on the page
        posspara: the indices of the long paragraphs/write-ups on the page
    '''
    out = []
    for header in headers:
        # distance to nearest long paragraph
        distpara = min(possparas - header,
                       key=lambda x: x if x > 0 else float('inf'))
        # distance to nearest address
        distaddr = min(addresses - header,
                       key=lambda x: x if x > 0 else float('inf'))
        out.append(distpara + distaddr)
    return np.array(out)


def onlyNumbers(teststr):
    re1 = re.compile('.*[0-9].*')
    re2 = re.compile('.*[a-z].*|.*[A-Z].*')
    if bool(re1.match(teststr)) and not re2.match(teststr):
        return True
    return False


# work on this... decide which header to return
def getoneheader(soup, out, paragraphs):
    page_title = soup.select("title")[0].get_text().encode(
        'ascii', 'ignore').strip()
    bkpt = 0
    # print page

    '''
    get the stuff in the page title before the 1st punctuation mark
    as usually if a restaurant's name is 'Bistrot Belhara' then the title
    of the page is 'Bistrot Belhara | Paris by Mouth - Mozilla Firefox'
    '''
    for i in range(len(page_title)):
        if page_title[i] in string.punctuation and page_title[i] != '\'':
            bkpt = i
            break

    page_title = page_title[0:bkpt].strip()
    print page_title

    # page_title in one of the titles or one of the titles in page_title
    lwr = [t.lower() for t in out]
    posstitle = [l for l in lwr if l in page_title.lower()]
    print posstitle
    if page_title.lower() in lwr or len(posstitle) != 0:
        if len(page_title) < len(posstitle[0]):
            return [page_title]
        else:
            return [posstitle[0]]


if __name__ == '__main__':
    url = raw_input("enter website to parse\n")
    # opener = urllib2.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    # response = opener.open(url)
    # page = response.read()
    # soup = BeautifulSoup(page, 'lxml')
    titles = getTitle(url)

    # print str(len(titles)) + " titles found on page!\n"

    # page_title = soup.select("title")[0].get_text()

    # lwr = [t.lower() for t in page_title]
    # if page_title.lower() in lwr:
    #     print "single page title " + page_title

    # for t in titles:
    #     print t
