from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import numpy as np
import urllib2
import string
import sys
import os
import re

stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

st = TreebankWordTokenizer()

def getTitle(url, addresses):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')

    for elem in soup.findAll(['script', 'style']):
        elem.extract()

    out = set()

    raw = soup.get_text().encode('ascii', 'ignore')
    paras = [p.strip() for p in raw.split('\n') if len(p.strip()) > 2]
    paradict = {}
    for i in range(len(paras)):
        if paras[i] not in paradict:
            paradict[paras[i]] = i

    # separate method for ladyironchef.com
    if 'ladyironchef' in url:
        tags = soup.findAll('span', {"style":"font-size: x-large;"})
        titles = []
        paradict = {}

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

        return titles

    # trip advisor has only a single place_name/page
    elif 'tripadvisor' in url:
        page_title = soup.findAll("title")[0].get_text().encode('ascii', 'ignore')

        for i in range(len(page_title)):
            if page_title[i] in string.punctuation:
                break

        page_title = page_title[0:i]
        return [page_title]

    else:

        # in a few rare cases the title of page can be under <Hn> inside the page also
        # header_titles = [soup.findAll('h'+str(i)) for i in range(1, 5)]
        # header_titles.append(soup.findAll('strong'))
        text = soup.findAll(re.compile('h[0-5]|strong'))

        # to select the elements which have the maximum number of common tags
        # text = max(header_titles)
        for title in text:
            str1 = title.get_text().encode('ascii', 'ignore')
            str1 = str1.replace('\t', '')
            str1 = str1.replace('\n', '')
            if len(str1) > 2 and (not onlyNumbers(str1)) and str1 in paras:
                print str1
                out.add(str1)

        # this implies that most probably page is a multi-place blog
        print addresses
        if len(addresses[0]) <= 3:
            onetitle = getoneheader(soup, out)
            return onetitle

        out = list(out)
        out = sorted(out, key=lambda x: paradict[x])
        return out

def onlyNumbers(teststr):
    re1 = re.compile('.*[0-9].*')
    re2 = re.compile('.*[a-z].*|.*[A-Z].*')
    if bool(re1.match(teststr)) and not re2.match(teststr):
        return True
    return False

def getoneheader(soup, out):
    page_title = soup.select("title")[0].get_text().encode('ascii', 'ignore').strip()
    bkpt = 0
    # print page
    for i in range(len(page_title)):
        if page_title[i] in string.punctuation and page_title[i] != '\'':
            bkpt = i
            break

    page_title = page_title[0:bkpt].strip()
    print page_title
    ## page_title in one of the titles or one of the titles in page_title
    lwr = [t.lower() for t in out]
    posstitle = [l for l in lwr if l in t.lower()]
    print posstitle
    if page_title.lower() in lwr or len(posstitle) != 0:
        if len(page_title) < len(posstitle[0]):
            return [page_title]
        else:
            return [posstitle[0]]

# if __name__ == '__main__':
#     url = raw_input("enter website to parse\n")
#     opener = urllib2.build_opener()
#     opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
#     response = opener.open(url)
#     page = response.read()
#     soup = BeautifulSoup(page, 'lxml')
#     titles = getTitle(url, [[], [], []])

#     print str(len(titles)) + " titles found on page!\n"

#     page_title = soup.select("title")[0].get_text()

#     lwr = [t.lower() for t in page_title]
#     if page_title.lower() in lwr:
#         print "single page title " + page_title

#     for t in titles:
#         print t