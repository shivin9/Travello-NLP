from nltk.tokenize import TreebankWordTokenizer
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
from scipy.fftpack import dct
import numpy as np
import urllib

st = TreebankWordTokenizer()

url = raw_input("enter website to parse\n")
soup = BeautifulSoup(urllib.urlopen(url).read(), 'lxml')

for elem in soup.findAll(['script', 'style']):
    elem.extract()

raw = soup.get_text().encode('ascii', 'ignore')
raw = raw.replace('\t', '')
paragraphs = [p.strip() for p in raw.split('\n') if len(p.strip()) > 2]
lens = [len(st.tokenize(p)) for p in paragraphs]

# lens = np.array([len(p) for p in paragraphs])
avg = np.average(lens)
disc = [l>avg for l in lens]

lensft = abs(dct(disc))
print paragraphs
print lens
plt.plot(disc)
plt.show()
plt.plot(lensft)
plt.show()
l = float(len(lens))
prd = np.argmax(lensft)
print (np.argsort(lensft)/l)*10