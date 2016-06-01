from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import numpy as np
import string
import urllib
import sys
import os
import re

stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

st = TreebankWordTokenizer()

def getTitle(url):
    out = set()
    soup = BeautifulSoup(urllib.urlopen(url).read(), 'lxml')

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
        header_titles = [soup.findAll('h'+str(i)) for i in range(1, 5)]
        header_titles.append(soup.findAll('strong'))
        # text = soup.findAll(re.compile('h[0-5]|strong'))

        # to select the elements which have the maximum number of common tags
        text = max(header_titles)
        for title in text:
            str1 = title.get_text().encode('ascii', 'ignore')
            str1 = str1.replace('\t', '')
            str1 = str1.replace('\n', '')
            out.add(str1)

        page_title = soup.select("title")[0].get_text().encode('ascii', 'ignore')
        for i in range(len(page_title)):
            if page_title[i] in string.punctuation and page_title[i] != '\'':
                break

        page_title = page_title[0:i].strip()
        print out
        lwr = [t.lower() for t in out]
        if page_title.lower() in lwr:
            return [page_title]

        return out

if __name__ == '__main__':
    url = raw_input("enter website to parse\n")
    titles = getTitle(url)
    print str(len(titles)) + " titles found on page!\n"
    page_title = soup.select("title")[0].get_text()

    lwr = [t.lower() for t in page_title]
    if page_title.lower() in lwr:
        print "single page title " + page_title

    for t in titles:
        print t