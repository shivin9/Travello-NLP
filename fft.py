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
std = np.std(lens)

disc = [l + 2*std > avg for l in lens]
print len(disc)
i = 0
j = len(disc) - 1

while disc[i] == 0:
    i+=1

while disc[j] == 0:
    j-=1
disc = disc[i:j+1]
print len(disc)
lensft = abs(dct(disc))
# print paragraphs
print lensft
plt.plot(lens[i:j+1])
plt.show()
plt.plot(lensft)
plt.show()
prd = np.argmax(lensft)
print (np.argsort(lensft)/l)*10