from nltk.tokenize import TreebankWordTokenizer
import datefinder
import random
import json
import re

st = TreebankWordTokenizer()

# this is the hard data which is used to create the feature vectors for a paragraph

# most-used street names
with open('./database/hard_data/streets.json', 'r') as f:
    streets = json.load(f)

# names of all states in all countries
with open('./database/hard_data/states.json', 'r') as f:
    states = json.load(f)

# names of all cities of the world
with open('./database/hard_data/cities.json', 'r') as f:
    cities = json.load(f)

# names of all the countries of the world
with open('./database/hard_data/countries.json', 'r') as f:
    countries = json.load(f)

# to simulate a real blog, we place non-address paragraphs taken from real
# blogs and append it below and above an address to simulate a real blog
with open('./database/hard_data/garbage', 'r') as f:
    garbage = f.read()

# a sample list of 1000 restaurants
with open('./database/hard_data/cafes', 'r') as f:
    cafes = f.read()

# regular expression for finding phone numbers
rephone = re.compile(
    r'\+[0-9][0-9]*|\([0-9]{3}\)|[0-9]{4} [0-9]{4}|([0-9]{3,4}[- ]){2}[0-9]{3,4}|[0-9]{10}')

# regular expression for finding numbers
renum = re.compile(r'(?i)^[a-z0-9][a-z0-9\- ]{4,8}[a-z0-9]$')

garbage = garbage.split('\n')
garbage = [g for g in garbage if g != '']

cafes = cafes.split('\n')
cafes = [c for c in cafes if c != '']

lengths1 = []
lengths2 = []

# the street names have a number indicating it's popularity eg. Road, Drive, Lane etc.
# are most popular. So we find the sum of all scores to normalize them later
summ = 0
for key in streets.keys():
    summ += streets[key]

summ = float(summ) / 3


def generate_data():
    '''
    This method creates synthetic blogs which have hierarchical addresses. Since the Wallmart
    dataset is highly regular ie. it is of the form :-
    address1;
    city, state, postalcode;
    country

    So to break this regularity as we can expect any type of format in real life blogs,
    we randomly omit the last 2 parts of the address ie. the phone number and the
    city, state and poscode part.

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    Used addresses of all 5800 Wallmart stores in US
    '''

    labels1 = []
    with open('./database/hard_data/walmart-full.json') as addrs:
        addrs = json.load(addrs)

    addresses_train = []
    print "generating hierarchical addresses..."
    for i in range(len(addrs)):
        temp = []
        y = []
        cnt = 0
        rnum = random.random()
        gnum1 = -1
        gnum2 = -1

        # for selecting the number of garbage texts above and below the address
        while gnum1 < 0 or gnum2 < 0:
            gnum1 = int(random.gauss(10, 5))
            gnum2 = int(random.gauss(10, 5))
        # gnum1 = 0
        # gnum2 = 0
        temp += random.sample(garbage, gnum1)
        y += [0] * gnum1

        # necessarily append the address1
        temp.append(addrs[i]['address']['address1'].encode('ascii', 'ignore'))
        cnt += 1
        if rnum > 0.05:
            temp.append(addrs[i]['address']['city'].encode('ascii', 'ignore') + ", " + addrs[i]['address'][
                        'state'].encode('ascii', 'ignore') + ", " + addrs[i]['address']['postalCode'].encode('ascii', 'ignore'))
            cnt += 1

            # dont put phone numbers in all cases as then it will learn that only
            if rnum > 0.6 and 'phone' in addrs[i]:
                temp.append(addrs[i]['phone'].encode('ascii', 'ignore'))
                cnt += 1
        y += [1] * cnt
        temp += random.sample(garbage, gnum2)
        y += [0] * gnum2
        labels1 += y
        lengths1.append(len(y))

        # for i in range(len(y)):
        #     print (temp[i], y[i])

        addresses_train.append(temp)

    data_vec = []

    for i in range(len(addresses_train)):
        if i % 100 == 0:
            print i
        data_vec += getdet(addresses_train[i])

    with open("./database/features/train1", "w") as f:
        print >> f, addresses_train

    with open("./database/features/labels1.py", "w") as f1:
        print >> f1, labels1

    with open("./database/features/lenghts1.py", "w") as f1:
        print >> f1, lengths1

    with open("./database/features/datavec1.py", "w") as f2:
        print >> f2, data_vec


def oneliners():
    '''
    This method creates synthetic blogs which have oneline addresses. Since the New-York
    Restaurants dataset is highly regular ie. it is of the form :-
    address1; city, state, postalcode; country

    So to break this regularity as we can expect any type of format in real life blogs,
    we randomly omit the last 2 parts of the address ie. the phone number and the
    city, state and poscode part.

    Parameters
    ----------

    Returns
    -------

    References
    ----------
    Used addresses of 6000 restaurants in New-York
    '''

    with open('./database/hard_data/us_rest1.json') as rests:
        rests = json.load(rests)

    print "generating one line addresses..."

    # randomly select data of 6000 restaurants from the json file
    randlist = random.sample(range(1, len(rests['data'])), 6000)
    one_line_addrs = []
    idx = 0

    # in the rests dictionary, the address portions are at the given indices
    order = [9, 11, 12, 13, 14]

    labels2 = []

    # for selecting the number of garbage texts above and below the address

    for idx in randlist:
        str1 = ""
        temp = []
        # print idx
        y1 = []
        rnum = random.random()

        gnum1 = -1
        gnum2 = -1
        while gnum1 <= 0 or gnum2 <= 0:
            gnum1 = int(random.gauss(10, 5))
            gnum2 = int(random.gauss(10, 5))

        temp += random.sample(garbage, gnum1)

        # concatenate gnum1 number of 0s to indicate gnum1 number of garbage paragraphs
        y1 += [0] * gnum1

        # here also we randomly remove some parts of the one-line addresses
        # this translates to chopping off some portion from the back of the order list
        ordd = order

        if rnum < 0.5:
            ordd = order[:-1]
        if rnum < 0.4:
            ordd = ordd[:-1]

        for od in ordd:
            part = rests['data'][idx][od]
            if part != None:
                str1 += part.encode("ascii", "ignore") + ", "

        # this is to capitalize the first letter of all the words
        str1 = str1.title()
        temp.append(str1)
        temp += random.sample(garbage, gnum2)

        # concatenate a 1 ie. address for this one-line address
        y1 += [1]

        # concatenate gnum2 number of 0s to indicate gnum2 number of garbage paragraphs
        y1 += [0] * gnum2
        lengths2.append(len(y1))
        labels2 += y1
        one_line_addrs.append(temp)

    data_vec = []

    for i in range(len(one_line_addrs)):
        if i % 100 == 0:
            print i
        data_vec += getdet(one_line_addrs[i])

    with open("./database/features/train2", "w") as f:
        print >> f, one_line_addrs

    with open("./database/features/labels2.py", "w") as f1:
        print >> f1, labels2

    with open("./database/features/lengths2.py", "w") as f1:
        print >> f1, lengths2

    with open("./database/features/datavec2.py", "w") as f2:
        print >> f2, data_vec


def getdet(data):
    '''
    This method is used to get all tha labels for a single synthetic blog
    It calls the getvec function for all the paragraphs inside it

    Parameters
    ----------
    data : A list of paragraphs which forms a synthetic blog

    Returns
    -------
    feature_vec : A list of shape=(n_paragraphs, 8)
    '''

    feature_vec = []
    for i in range(len(data)):
        feature_vec.append(getvec([data[i]]))
    return feature_vec


def getvec(lines):
    '''
    features:
        number of streets(0), cities(1), states(2), countries(3): INT
        sum of weights of the streets: FLOAT (4)
        has phone number?: 0/1 (5)
        zip codes?: 0/1 (6)
        length of paragraph: INT (7)
        has date?: 0/1 (8)

    This method calculates the feature vector for a single paragraph using the above
    features

    Parameters
    ----------
    data : A list of paragraphs which forms a synthetic blog

    Returns
    -------
    vec : A list length 8

    '''
    vec = [0]*9
    for line in lines:
        phnum = len(rephone.findall(line))
        nums = len(renum.findall(line))
        numterm = 0

        for terms in st.tokenize(line):
            numterm+=1
            # terms = terms.lower()
            if terms.lower() in streets:
                vec[0] += 1
                vec[4] += streets[terms.lower()]/float(summ)

            if terms in states:
                # state names are biased towards US and Australia addresses
                # therefore we don't add their weights
                vec[1] += 1

            if terms in cities:
                vec[2] += 1

            if terms in countries:
                vec[3] += 1

        vec[5] = phnum
        vec[6] = nums
        vec[7] = numterm

        matches = datefinder.find_dates(line, strict=True)
        try:
            for match in matches:
                vec[8] = 1
                break
        except:
            pass
    return vec


if __name__ == '__main__':
    generate_data()
    oneliners()
