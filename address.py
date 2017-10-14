from nltk.tokenize import TreebankWordTokenizer
from sklearn.cluster import KMeans
import numpy as np
import datefinder
import json
import sys
import re
from utils import parsePage, getData, getScores
from models import rulEx

sys.path.insert(0, './database/')
st = TreebankWordTokenizer()

with open('./database/hard_data/streets.json', 'r') as f:
    streets = json.load(f)
with open('./database/hard_data/states.json', 'r') as f:
    states = json.load(f)
with open('./database/hard_data/cities.json', 'r') as f:
    cities = json.load(f)
with open('./database/hard_data/countries.json', 'r') as f:
    countries = json.load(f)


def getAddress(url, predictors):
    '''
    Finds all the addresses on the web-page

    Parameters
    ----------
    url : The url of the page

    predictors : a list of tuples which are like (parameters, model)
        Here parameters is a dictionary of the hyper-parameters of the model

    Returns
    -------
    final : A list of lists, where every list contains the paragraph which are the
        part of the same address.
    '''

    soup, paras, paradict = parsePage(url)
    # print soup
    addresses = []

    if 'tripadvisor' in url:
        final = TripAdAddr(soup)

    else:
        results = set()

        for params, pred in predictors:
            # get the feature vectors for the text on the web-page as required by the
            X = getData(paras, params['NUM_FEATURES'], params[
                'BATCH_SIZE'], SEQ_LENGTH=params['SEQ_LENGTH'])
            res = pred(X).flatten()
            addrs = getLabels(res, paras, params['NUM_CLUST'])

            # take the intersection of the results extracted by the classifiers...
            # success depends heavily on the ability of the classifiers to find all the addresses
            results = results.intersection(addrs)
            #print getScores(pred, paras, params)

        # the final address extractor is the hard coded rule-based function which works when
        # there are telephone numbers in the address
        results = results.union(rulEx(paras))

        # to align the addresses based on their position on the page
        addresses = sorted(results, key=lambda x: x[1])
        final = accuAddr(addresses)

    # print final
    return final


def TripAdAddr(soup):
    '''
    Find all the addresses on a TripAdvisor page.
    Note : TripAdvisor has only one point of interest per page... hence no loops here

    Parameters
    ----------
    soup : The soup object corresponding to the TravelAdvisor page

    Returns
    -------
    final : A list of lists, where every list contains the index of paragraph which are the
        part of the same address.
    '''

    # remove the trailing spaces and the extra commas
    strt = soup.findAll("span", {"class": 'street-address'})[0].get_text().encode('ascii', 'ignore').strip().strip(',')

    loc = soup.findAll("span", {"class": 'locality'})[0].get_text().encode('ascii', 'ignore').strip().strip(',')

    count = soup.findAll("span", {"class": 'country-name'})[0].get_text().encode('ascii', 'ignore').strip().strip(',')

    return [[strt + ", " + loc + ", " + count]]


def accuAddr(addresses):
    '''
    After identifying the individual paragraphs which are of address type, we need to
    combine them into a single structure as addresses can be hierarchical

    Parameters
    ----------
    url : A list numbers addresses where each entry is the index of an address

    Returns
    -------
    final : A list of lists, where every list contains the index of paragraph which are the
        part of the same address.
    '''

    i = 0
    final = []
    while i < len(addresses):
        accued = [addresses[i][0]]
        while i + 1 < len(addresses) and (addresses[i + 1][1] - addresses[i][1]) <= 2:
            accued += [addresses[i + 1][0]]
            i += 1
        # if not hasdate(accued):
        final += [accued]
        i += 1
    return final


def hasdate(address):
    '''
    Function for removing dates from addresses. Dates were still coming in the addresses,
    so I deecided to manually filter them out

    Parameters
    ----------
    address :

    predictors : a list of tuples which are like (parameters, model)
        Here parameters is a dictionary of the hyper-parameters of the model

    Returns
    -------
    final : A list of lists, where every list contains the index of paragraph which are the
        part of the same address.
    '''

    str1 = " ".join(address)
    matches = datefinder.find_dates(str1, strict=True)
    for match in matches:
        return True
    return False


def getLabels(scores, paragraphs, NUM_CLUST=2):
    '''
    This function decides which paragraphs are actually addresses given their probabilities
    as returned by the classifier.
    We cant hardcode a certain threshold for deciding addresses as it may depend from page
    to page so we therefore segregate the paragraphs into 2 clusters with their score as the key.

    Parameters
    ----------
    scores : A numpy array, The scores as assigned to the paragraphs by the classifier

    paragraphs : A list of all paragraphs on the webpage

    NUM_CLUST : The number of clusters into which we want to segregate the paragraphs...
        Default value is 2 as it works well in practice.

    Returns
    -------
    addresses : A list of tuples, The tuple elements of the tuple are the paragraph
        and it's index in the paras list.
    '''

    scores = scores.reshape(-1, 1)
    est = KMeans(n_clusters=NUM_CLUST)
    est.fit(scores)
    labels = est.labels_

    # A dictionary to keep track of the paragraphs and which clusters they belong to
    dict = {}
    for i in range(NUM_CLUST):
        dict[i] = []

    # this is the index of the paragraph which is the 'best address'...
    # if it's score is less than 0.5(whimsically chosen) then we say
    # that there are no addresses on the page
    bestaddr = np.argmax(scores)
    if scores[bestaddr] < 0.5:
        return []

    # segregate the paragraphs into 'NUM_CLUST' clusters
    for index, para in enumerate(paragraphs):
        dict[labels[index]].append((para, index))

    # the required addresses belong to the cluster to which the
    # 'best address' belongs to
    addresses = dict[labels[bestaddr]]

    return addresses


# new_address tries to do both ie. hierarchical and one-line addresses in one go
# this function is depreciated...
def new_address(text):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 2]
    tokked = [st.tokenize(p) for p in paragraphs]
    lens = [len(tokked) for p in paragraphs]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(
        r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}|[0-9]{10}')
    # print paragraphs
    # print lens
    possible_addresses = []
    idx = 0
    while idx < len(paragraphs):
        # first filter the paragraphs by phone number
        if bool(regexp.search(paragraphs[idx])):
            poss = []
            poss.append(paragraphs[idx])
            temp = idx - 1
            # print paragraphs[idx]
            # go back till we are seeing an address
            while isAddr(paragraphs[temp]) and lens[temp] < 10:
                poss.append(paragraphs[temp])
                temp -= 1

            if len(poss) <= 15:
                possible_addresses.append(poss[::-1])
                idx += 1

        else:
            # some random number for max. length of an address
            poss = []
            if lens[idx] <= 20:
                if isAddr(paragraphs[idx]):
                    temp = idx
                    while isAddr(paragraphs[temp]) or bool(regexp.search(paragraphs[temp])):
                        if lens[temp] >= 10:
                            break
                        poss.append(paragraphs[temp])
                        temp += 1
                        if temp >= len(paragraphs):
                            break
                    # address less than 10 lines
                    if len(poss) < 10:
                        possible_addresses.append(poss)
                        idx = temp
        idx += 1
    return possible_addresses


# use ML techniques to fix the score increments
# this is also depreciated
def isAddr(test_addr):
    score = 0
    numterm = 0
    for terms in st.tokenize(test_addr):
        numterm += 1
        # terms = terms.lower()
        if terms in states:
            # print "state " + terms + " found!"
            score += 1
        if terms.lower() in streets:
            # print "street " + terms + " found!"
            score += 3
        if terms in cities:
            # print "city " + terms + " found!"
            score += 1
        if terms in countries:
            # print "country " + terms + " found!"
            score += 1
    return float(score) / numterm > 0.4
