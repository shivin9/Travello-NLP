from nltk.tokenize import TreebankWordTokenizer
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
from scipy.fftpack import dct
import numpy as np
import urllib2

st = TreebankWordTokenizer()

def fft(url):
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/50.0.2661.102 Chrome/50.0.2661.102 Safari/537.36')]
    response = opener.open(url)
    page = response.read()
    soup = BeautifulSoup(page, 'lxml')

    for elem in soup.findAll(['script', 'style']):
        elem.extract()

    raw = soup.get_text().encode('ascii', 'ignore')
    raw = raw.replace('\t', '')
    paragraphs = [p.strip() for p in raw.split('\n') if len(p.strip()) > 2]
    lens = [len(st.tokenize(p)) for p in paragraphs]

    lens = np.array([len(p) for p in paragraphs])
    avg = np.average(lens)
    std = np.std(lens)
    print std
    disc = [l > (avg + 2*std) for l in lens]
    # i = 0
    # j = len(disc) - 1

    # while disc[i] == 0:
    #     i+=1

    # while disc[j] == 0:
    #     j-=1
    # disc = disc[i:j+1]
    # print sum(disc)
    lensft = abs(dct(lens))
    print (sum(lensft[1:10])/float(len(lens)))*10
    plt.plot(lens)
    plt.show()
    plt.plot(lensft)
    plt.show()
    prd = np.argmax(lensft)

    print (np.argsort(lensft)/float(len(lens)))*10

def doubleMADsfromMedian(y,thresh=30):
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y<=m])
    right_mad = np.median(abs_dev[y>=m])
    y_mad = np.zeros(len(y))
    y_mad[y < m] = left_mad
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev/y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh

while(1):
    url = raw_input("enter website to parse\n")
    fft(url)