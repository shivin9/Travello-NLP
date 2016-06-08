import random
import json

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

garbage = garbage.split('\n')
garbage = [g for g in garbage if g!='']
print len(garbage)
labels = []
def generate_data():
    with open('./database/walmart-full.json') as addrs:
        addrs = json.load(addrs)

    addresses_train = []

    for i in range(len(addrs)):
        temp = []
        y = []
        cnt = 0
        print i
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
        labels.append(y)
        addresses_train.append(temp)

    with open("./database/train1", "w") as f:
        print >> f, addresses_train

    with open("./database/labels1", "w") as f1:
        print >> f1, labels

    with open('./database/us_rest1.json') as rests:
        rests = json.load(rests)

    randlist = random.sample(range(1, len(rests['data'])), 6000)
    one_line_addrs = []
    idx = 0
    order = [9, 11, 12, 13, 14]

    rnum = random.random()

    # for selecting the number of garbage texts above and below the address
    gnum1 = int(random.gauss(10, 5))
    gnum2 = int(random.gauss(10, 5))
    temp += random.sample(garbage, gnum1)
    y += [0]*gnum1

    for idx in randlist:
        str1 = ""
        print idx
        # dont put phone numbers in all cases as then it will learn that only

        for ordd in order:
            part = rests['data'][idx][ordd]
            if part != None:
                str1+= part.encode("ascii", "ignore")+", "
        one_line_addrs.append([str1])
        cnt+=1

    with open("train2", "w") as f:
        print >> f, one_line_addrs

def isAddr(test_addr):
    score = 0
    numterm = 0
    for terms in st.tokenize(test_addr):
        numterm+=1
        terms = terms.lower()
        if terms in states:
            print "state " + terms + " found!"
            score+=1
        if terms.lower() in streets:
            print "street " + terms + " found!"
            score+=1
        if terms in cities:
            print "city " + terms + " found!"
            score+=1
        if terms in countries:
            print "country " + terms + " found!"
            score+=1
    return float(score)/numterm > 0.4
generate_data()
