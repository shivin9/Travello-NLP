from nltk.tokenize import TreebankWordTokenizer
from sklearn.cluster import KMeans
import numpy as np
import datefinder
import json
import sys
import re
from utils import parsePage, getData

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
        predictors: it's a list of prediction
        functions like RNN, LSTM BoostedRNN etc.
    '''
    soup, paras, paradict = parsePage(url)
    addresses = []

    if 'tripadvisor' in url:
        final = TripAdAddr(soup)

    else:
        results = set()
        for params, pred in predictors:
            # pred(X, paras) for RULE based classifier
            X = getData(paras, params['NUM_FEATURES'], params[
                'BATCH_SIZE'], SEQ_LENGTH=params['SEQ_LENGTH'])
            res = pred(X).flatten()
            results = results.union(getLabels(res, paras, params['NUM_CLUST']))

        addresses = sorted(results, key=lambda x: x[1])
        # print addresses
        final = accuAddr(addresses)

    print final
    return final
    # raw = soup.get_text().encode('ascii', 'ignore')
    # raw = raw.replace('\t', '')
    # hier_addr = new_address(raw)
    # print str(len(hier_addr)) + " addresses found!"
    # print hier_addr
    # # direct_addr = direct_address(raw)
    # # print direct_addr
    # return [hier_addr]


def TripAdAddr(soup):
    # remove the trailing spaces and the extra commas
    strt = soup.findAll("span", {"class": 'street-address'})[0].get_text().encode('ascii', 'ignore').strip().strip(',')

    loc = soup.findAll("span", {"class": 'locality'})[0].get_text().encode('ascii', 'ignore').strip().strip(',')

    count = soup.findAll("span", {"class": 'country-name'})[0].get_text().encode('ascii', 'ignore').strip().strip(',')

    return [[strt + ", " + loc + ", " + count]]


def accuAddr(addresses):
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


# function for removing dates from addresses
def hasdate(address):
    str1 = " ".join(address)
    matches = datefinder.find_dates(str1, strict=True)
    for match in matches:
        return True
    return False


# new_address tries to do both ie. hierarchical and one-line addresses in
# one go
def new_address(text):
    paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 2]
    tokked = [st.tokenize(p) for p in paragraphs]
    lens = [len(tokked) for p in paragraphs]
    lens = [len(st.tokenize(p)) for p in paragraphs]
    regexp = re.compile(
        r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}|[0-9]{10}')
    # print paragraphs
    print lens
    possible_addresses = []
    idx = 0
    while idx < len(paragraphs):
        # first filter the paragraphs by phone number
        if bool(regexp.search(paragraphs[idx])):
            poss = []
            poss.append(paragraphs[idx])
            temp = idx - 1
            print paragraphs[idx]
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


def getLabels(res, paras, NUM_CLUST):
    res = res.reshape(-1, 1)
    est = KMeans(n_clusters=NUM_CLUST)
    est.fit(res)
    labels = est.labels_
    dict = {}

    for i in range(NUM_CLUST):
        dict[i] = []

    bestaddr = np.argmax(res)
    if res[bestaddr] < 0.5:
        return []

    for i in range(len(paras)):
        dict[labels[i]].append((paras[i], i))

    return dict[labels[bestaddr]]

# if __name__ == '__main__':
#     while 1:
#         try:
#             url = raw_input("enter website to parse\n")
#         except:
#             print "invalid url"
#         parsepage(url)
