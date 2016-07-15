from utils import parsePage, LongParas
import numpy as np
import string
import re

NUM_CLUSTERS = 2


def getTitle(url, addresses):
    '''
    Finds all the titles or headings on the webpage. We use locality arguments to
    find the titles which are actually the name of a point of interest

    Parameters
    ----------
    url : The url of the page

    addresses : This is a list addresses which were extracted by the address extractor

    Returns
    -------
    headers : A list of all the 'possible' place names on the webpage. No guarantees!
    But recall is 100%, precision is not ie. you get all the gold but along with
    some mud as well
    '''

    soup, paragraphs, paradict = parsePage(url)
    lens = [len(p) for p in paragraphs]

    # Note that although the functions return the headers and addresses etc. in
    # text, internally they play around with their indices as it is easier to apply
    # locality arguments on them
    headerIndices = []

    # separate method for ladyironchef.com
    if 'ladyironchef' in url:
        headerIndices = LICTitle(soup, paradict)

    # trip advisor has only a single place_name/page
    elif 'tripadvisor' in url:
        headerIndices = TripAdTitle(soup, paradict)

    # there are blogs and pages which have only one point of interest, so for
    # them we have a special function which for now just returns back the page title
    elif len(addresses) <= 3:
        # onetitle = getoneheader(soup, possibleHeaders, paragraphs)
        headerIndices = [-1]

    # this implies that most probably page is a multi-place blog
    else:
        possibleHeaders = GenPage(soup, paradict)

        # get the long write-ups about the headings. they will help us find the 'real'
        # headers as 'real' headers will have a long write-up about that place nearby
        # most probably to it's down
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

    headers = []
    for idx in headerIndices:
        headers.append(paragraphs[idx])

    return headers


def LICTitle(soup, paradict):
    '''
    Special function for LadyIronChef as all the headings are under the
    span tag.

    Parameters
    ----------
    soup : The soup of that page

    paradict : The dictionary mapping paragraphs to their indices

    Returns
    -------
    headers : A list of indices of the headers
    '''
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

    headers = [paradict[t] for t in titles]
    return headers


def TripAdTitle(soup, paradict):
    '''
    Special function for TripAdvisor as it has only one point of interest per page.
    The title is usually like

    Rhubarb, Singapore - Chinatown - Restaurant Reviews, Phone Number &amp; Photos - TripAdvisor

    And therefore we break the title of the page by punctuation marks and return
    the first segment


    Parameters
    ----------
    soup : The soup of that page

    paradict : The dictionary mapping paragraphs to their indices

    Returns
    -------
    headers : A list of indices of the headers
    '''

    page_title = soup.findAll("title")[0].get_text().encode('ascii', 'ignore')

    for i in range(len(page_title)):
        if page_title[i] in string.punctuation:
            break

    page_title = page_title[0:i]
    return [paradict[page_title]]


def GenPage(soup, paradict):
    '''
    The general headers extractor function which finds the possible headers in a webpage
    by extracting all those paragraphs which are in <Hn> or <strong> tags
    Further filters need to be still applied to remove bogus headers

    Parameters
    ----------
    soup : The soup of that page

    paradict : The dictionary mapping paragraphs to their indices

    Returns
    -------
    possheaders : A list of the indices of all the possible headers
    '''

    headings = soup.findAll(re.compile('h[0-5]|strong'))
    possheaders = set()
    # to select the elements which have the maximum number of common tags-- failed idea
    # strip the string of waste space from the sides

    for title in headings:
        head = title.get_text().encode('ascii', 'ignore')
        head = head.replace('\t', '')
        head = head.replace('\n', '')
        head = head.strip()
        # sometimes the head was found to be not in the paragraphs list
        # therefore this check was added, also many a times only numbers were getting
        # found so have removed them too
        if len(head) > 2 and (not onlyNumbers(head)) and head in paradict:
            possheaders.add(paradict[head])

    # return the headers in a sorted order
    possheaders = sorted(list(possheaders))
    return possheaders


def getHeadFeatures(headers, addresses, possparas):
    '''
    headers: the indices of the possible headers on the page
    addresses: the indices of the first line of address on the page
    posspara: the indices of the long paragraphs/write-ups on the page
    we use locality arguments with addresses also. Headers with an address
    nearby and that too to it's bottom is authentic as this is what the
    general structure of most blogs and webpages tends to be

    Parameters
    ----------
    headers : Indices of possible headers

    addresses : A list of indices of the first line of an address which for one-liners
        is the address itself and for hierarchical addresses is the first line

    possparas : A list of indices of long paragraphs ie. write-ups about a place of interest

    Returns
    -------
    out : A list numbers which act as the feature values for every possible header.
        It is the sum of distances of the header to the nearest write-up and
        the address. Positive value means that the headers is likely to be authentic.
    '''
    out = []
    for header in headers:
        distpara = min(possparas - header,
                       key=lambda x: x if x > 0 else float('inf'))
        distaddr = min(addresses - header,
                       key=lambda x: x if x > 0 else float('inf'))
        out.append(distpara + distaddr)
    out = np.array(out)
    return out


def onlyNumbers(teststr):
    '''
    Tests whether a string is only numbers

    Parameters
    ----------
    teststr : The string which is to be tested

    Returns
    -------
    A boolean variable True/False
    '''
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
