from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import numpy as np
import string
import urllib2
import sys
import os
import re

stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

st = TreebankWordTokenizer()

def getTitle(url):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')
    out = set()
    # separate method for ladyironchef.com
    if 'ladyironchef' in url:
        tags = soup.findAll('span', {"style":"font-size: x-large;"})
        titles = []
        for tag in tags:
            name = tag.get_text().encode('ascii', 'ignore')
            titles.append(name)
        return titles

    # trip advisor has only a single place_name/page
    elif 'tripadvisor' in url:
        page_title = soup.findAll("title")[0].get_text().encode('ascii', 'ignore')

        for i in range(len(page_title)):
            if page_title[i] in string.punctuation:
                break

        page_title = page_title[0:i]
        return [page_title]
    # clean the retrieved text
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
            if len(str1) > 2:
                out.add(str1)

        page_title = soup.select("title")[0].get_text().encode('ascii', 'ignore').strip()
        bkpt = 0
        # print page
        for i in range(len(page_title)):
            if page_title[i] in string.punctuation and page_title[i] != '\'':
                print page_title[i]
                bkpt = i
                break

        print bkpt
        page_title = page_title[0:bkpt].strip()

        print out
        print page_title
        lwr = [t.lower() for t in out]
        if page_title.lower() in lwr:
            return [page_title]

        return out

if __name__ == '__main__':
    url = raw_input("enter website to parse\n")
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')
    titles = getTitle(url)

    print str(len(titles)) + " titles found on page!\n"

    page_title = soup.select("title")[0].get_text()

    lwr = [t.lower() for t in page_title]
    if page_title.lower() in lwr:
        print "single page title " + page_title

    for t in titles:
        print t