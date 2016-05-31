from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import numpy as np
import urllib
import sys
import os
import re

stagger = StanfordNERTagger('/home/shivin/Documents/Travello-NLP/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', '/home/shivin/Documents/Travello-NLP/stanford-ner/stanford-ner.jar', encoding='utf-8')

st = TreebankWordTokenizer()

def getTitle(url, soup):
    out = set()

    # separate method for ladyironchef.com
    if 'ladyironchef' in url:
        pass
    # clean the retrieved text
    else:
        text = soup.findAll(re.compile('h[0-5]|strong'))
        for title in text:
            str1 = title.get_text()
            str1 = str1.replace('\t', '')
            str1 = str1.replace('\n', '')
            # if str1 in page_title:
            #     return [str1]
            out.add(str1)

    return out

if __name__ == '__main__':
    url = raw_input("enter website to parse\n")
    soup = BeautifulSoup(urllib.urlopen(url).read(), 'lxml')

    titles = getTitle(url, soup)
    print str(len(titles)) + " titles found on page!\n"
    page_title = soup.select("title")[0].get_text()

    lwr = [t.lower() for t in page_title]
    if page_title.lower() in lwr:
        print "single page title " + page_title

    for t in titles:
        print t