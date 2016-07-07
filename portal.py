from flask import render_template
from models import getModel
from title import getTitle
from address import getAddress
from utils import consolidateStuff, getImg
from flask import request
from flask import Flask

app = Flask(__name__)

params = {'GRAD_CLIP': 100,
          'NAME': 'RNN',
          'SEQ_LENGTH': 1,
          'NUM_EPOCHS': 20,
          'LEARNING_RATE': 0.01,
          'N_HIDDEN': 512,
          'PRINT_FREQ': 5,
          'NUM_FEATURES': 9,
          'BATCH_SIZE': 512,
          'NUM_CLUST': 3}

paramsold = {'BATCH_SIZE': 512,
             'GRAD_CLIP': 100,
             'LEARNING_RATE': 0.01,
             'NAME': 'RNN',
             'NUM_CLUST': 3,
             'NUM_EPOCHS': 20,
             'NUM_FEATURES': 8,
             'N_HIDDEN': 512,
             'PRINT_FREQ': 5,
             'SEQ_LENGTH': 1,
             'TYPE': 'new feature1'}

paramslstm = {'BATCH_SIZE': 512,
              'GRAD_CLIP': 100,
              'LEARNING_RATE': 0.01,
              'NAME': 'LSTM',
              'NUM_CLUST': 3,
              'NUM_EPOCHS': 10,
              'NUM_FEATURES': 8,
              'N_HIDDEN': 512,
              'PRINT_FREQ': 5,
              'SEQ_LENGTH': 4,
              'TYPE': 'new feature1'}

try:
    # print paramsold
    rnnModelold = getModel(paramsold, "rnnmodel-old")
except:
    print "couldn't create the model... enter a valid filename"

try:
    # print paramsold
    rnnModel = getModel(params, "newest")
except:
    print "couldn't create the model... enter a valid filename"


try:
    # print paramslstm
    lstmmodel = getModel(paramslstm, "lstmodel-old")
except:
    print "couldn't create the model... enter a valid filename"


@app.route('/')
def index():
    return render_template("form1.html")


@app.route('/', methods=['POST'])
def post_form():
    url = request.form['text']
    addresses = getAddress(url, [(paramsold, rnnModelold), (params, rnnModel)])
    titles = getTitle(url, addresses)
    images = getImg(url)
    str_to_return = consolidateStuff(url, titles, addresses, images)
    print str_to_return
    str_to_return = str_to_return.replace('\n', '<br>')
    return str_to_return

if __name__ == '__main__':
    app.run()
