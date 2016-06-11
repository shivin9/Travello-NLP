from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import StanfordNERTagger
from stemming.porter2 import stem
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing
import pandas as pd
import numpy as np
import string
import urllib
import random
import json
import sys
import os
import re

st = TreebankWordTokenizer()

with open('./database/streets.json', 'r') as f:
    streets = json.load(f)
with open('./database/states.json', 'r') as f:
    states = json.load(f)
with open('./database/cities.json', 'r') as f:
    cities = json.load(f)
with open('./database/countries.json', 'r') as f:
    countries = json.load(f)
with open('./database/garbage', 'r') as f:
    garbage = f.read()

reph = re.compile(r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}|[0-9]{10}')
renum = re.compile(r'[0-9]+')

garbage = garbage.split('\n')
garbage = [g for g in garbage if g!='']
print len(garbage)
labels1 = []
labels2 = []
summ = 0
for key in streets.keys():
    summ += streets[key]

def generate_data():
    with open('./database/walmart-full.json') as addrs:
        addrs = json.load(addrs)

    addresses_train = []
    print "generating hierarchical addresses..."
    for i in range(len(addrs)):
        temp = []
        y = []
        cnt = 0
        rnum = random.random()
        gnum1 = -1
        # for selecting the number of garbage texts above and below the address
        while gnum1 < 0 or gnum2 < 0:
            gnum1 = int(random.gauss(10, 5))
            gnum2 = int(random.gauss(10, 5))

        temp += random.sample(garbage, gnum1)
        y += [0]*gnum1

        # necessarily append the address1
        temp.append(addrs[i]['address']['address1'].encode('ascii', 'ignore'))
        cnt += 1
        if rnum > 0.05:
            temp.append(addrs[i]['address']['city'].encode('ascii', 'ignore')+", "+addrs[i]['address']['state'].encode('ascii', 'ignore')+", "+addrs[i]['address']['postalCode'].encode('ascii', 'ignore'))
            cnt += 1

            # dont put phone numbers in all cases as then it will learn that only
            if rnum > 0.4 and 'phone' in addrs[i]:
                temp.append(addrs[i]['phone'].encode('ascii', 'ignore'))
                cnt += 1
        y += [1]*cnt
        temp += random.sample(garbage, gnum2)
        y += [0]*gnum2
        labels1.append(y)
        addresses_train.append(temp)

    # buffer address...
    # addresses_train.append([""])
    data_vec = []

    for i in range(len(addresses_train)):
        data_vec.append(getdet(addresses_train[i]))


    with open("./database/train1", "w") as f:
        print >> f, addresses_train

    with open("./database/labels1", "w") as f1:
        print >> f1, labels1

    with open("./database/datavec1", "w") as f2:
        print >> f2, data_vec

def oneliners():
    with open('./database/us_rest1.json') as rests:
        rests = json.load(rests)
    print "generating one line addresses..."

    randlist = random.sample(range(1, len(rests['data'])), 6000)
    one_line_addrs = []
    idx = 0
    order = [9, 11, 12, 13, 14]

    # for selecting the number of garbage texts above and below the address

    for idx in randlist:
        str1 = ""
        temp = []
        # print idx
        y1 = []
        rnum = random.random()

        gnum1 = -1
        while gnum1<=0 or gnum2 <= 0:
            gnum1 = int(random.gauss(10, 5))
            gnum2 = int(random.gauss(10, 5))
        temp += random.sample(garbage, gnum1)
        y1 += [0]*gnum1
        ordd = order
        if rnum < 0.5:
            ordd = order[:-1]
        if rnum < 0.4:
            ordd = ordd[:-1]

        for od in ordd:
            part = rests['data'][idx][od]
            if part != None:
                str1+= part.encode("ascii", "ignore")+", "
        str1 = str1.title()
        temp.append(str1)
        temp += random.sample(garbage, gnum2)
        # print temp
        y1 += [1]
        y1 += [0]*gnum2

        labels2.append(y1)
        one_line_addrs.append(temp)

    data_vec = []

    for i in range(len(one_line_addrs)):
        data_vec.append(getdet(one_line_addrs[i]))

    with open("./database/train2", "w") as f:
        print >> f, one_line_addrs

    with open("./database/labels2", "w") as f1:
        print >> f1, labels2

    with open("./database/datavec2", "w") as f2:
        print >> f2, data_vec


def getdet(data):
    data.append("")
    feature_vec = []
    for i in range(len(data) - 1):
        feature_vec.append(getvec([data[i], data[i+1]]))
    return feature_vec

def getvec(lines):
    '''
        features:
            number of streets(0), cities(1), states(2), countries(3) of current and next if present
            sum of weights of the streets(4)
            has phone number?(5)
            number of numbers(6)
            length of paragraph(7)
            result of prev. state if any
    '''
    vec = [0]*16
    par = 0
    for line in lines:
        phnum = len(reph.findall(line))
        nums = len(renum.findall(line))
        numterm = 0

        for terms in st.tokenize(line):
            numterm+=1
            # terms = terms.lower()
            if terms.lower() in streets:
                vec[0 + 8*par] += 1
                vec[4 + 8*par] += streets[terms.lower()]/float(summ)

            if terms in states:
                vec[1 + 8*par] += 1

            if terms in cities:
                vec[2 + 8*par] += 1

            if terms in countries:
                vec[3 + 8*par] += 1

            vec[5 + 8*par] = phnum
            vec[6 + 8*par] = nums
            vec[7 + 8*par] = numterm
            par = 1

    return vec

generate_data()
oneliners()