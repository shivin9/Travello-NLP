from flask import render_template
from models import getModel
import time
from title import getTitle
from address import getAddress
from utils import consolidateStuff, getImg
from flask import request
from flask import Flask

app = Flask(__name__)

# these parameter dictionaries show the parameters according to which we a
# classifer

params = {'GRAD_CLIP': 100,
          'NAME': 'RNN',
          'SEQ_LENGTH': 1,
          'NUM_EPOCHS': 20,
          'LEARNING_RATE': 0.01,
          'N_HIDDEN': 512,
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
             'SEQ_LENGTH': 1,
             'TYPE': '1st classifier'}

paramslstm = {'BATCH_SIZE': 512,
              'GRAD_CLIP': 100,
              'LEARNING_RATE': 0.01,
              'NAME': 'LSTM',
              'NUM_CLUST': 3,
              'NUM_EPOCHS': 10,
              'NUM_FEATURES': 8,
              'N_HIDDEN': 64,
              'SEQ_LENGTH': 4,
              'TYPE': '2nd classifier'}

# here in this example we are fetching a pre-trained model. Note
# that all the saved models are in the ./models directory
try:
    # print paramsold
    rnnModelold = getModel(paramsold, "rnnmodel-old")
except:
    print "couldn't create the model... please correct the error"

try:
    # print paramsold
    rnnModel = getModel(params, "newest")
except:
    print "couldn't create the model... enter a valid filename"

try:
    # print paramslstm
    lstmmodel = getModel(paramslstm, "lstmodel-old")
except:
    print "couldn't create the model... please correct the error"


@app.route('/')
def index():
    '''
    This function loads the html form which is saved in the ./templates folder
    by the name of form.html

    Parameters
    ----------

    Returns
    -------
        It returns a rendred template of the html form
    '''
    return render_template("form1.html")


@app.route('/', methods=['POST'])
def post_form():
    '''
    This method is to access/test the package via the web portal which
    runs at the address ip:5000
    '''
    url = request.form['text']
    start_time = time.time()

    # an ensemble of 3 Neural Network models
    try:
        addresses = getAddress(
            url, [(paramsold, rnnModelold), (paramslstm, lstmmodel)])
        print("addresses took {:.3f}s".format(time.time() - start_time))
    except:
        return "{error: Cant retrieve address}"

    t1 = time.time()

    try:
        titles = getTitle(url, addresses)
        print("titles took {:.3f}s".format(time.time() - t1))
    except:
        return "{error: Cant retrieve titles}"

    t2 = time.time()

    try:
        images = getImg(url)
        print("images took {:.3f}s".format(time.time() - t2))
    except:
        return "{error: Cant retrieve images}"

    t3 = time.time()
    try:
        str_to_return = consolidateStuff(url, titles, addresses, images)
        print("consolidation took {:.3f}s".format(time.time() - t3))
    except:
        return "{error: Cant consolidate the final information}"

    print str_to_return
    str_to_return = str_to_return.replace('\n', '<br>')
    return str_to_return


@app.route('/json-data')
def json_data():
    '''
    This method is similar to the previous one but note that is is hosted at
    /json-data/ for the travello application to query it. No webpage is rendered
    like the previous method
    '''

    url = request.args.get('url', 2)
    print url

    start_time = time.time()
    # an ensemble of 2 Neural Network models
    try:
        addresses = getAddress(
        url, [(paramsold, rnnModelold), (params, rnnModel), (paramslstm, lstmmodel)])
       	print("obtained addresses")
	print("addresses took {:.3f}s".format(time.time() - start_time))
    except:
        return "{error: Cant retrieve address - json_data}"

    t1 = time.time()

    try:
        titles = getTitle(url, addresses)
        print("titles took {:.3f}s".format(time.time() - t1))
    except:
        return "{error: Cant retrieve titles}"

    t2 = time.time()

    try:
        images = getImg(url)
        print("images took {:.3f}s".format(time.time() - t2))
    except:
        return "{error: Cant retrieve images}"

    t3 = time.time()
    try:
        str_to_return = consolidateStuff(url, titles, addresses, images)
        print("consolidation took {:.3f}s".format(time.time() - t3))
    except:
        return "{error: Cant consolidate the final information}"

    print str_to_return
    return str_to_return


if __name__ == '__main__':
    app.run()

    # uncomment these lines to debug your error
    # url = "http://www.ladyironchef.com/2017/10/birdfolks-singapore/"
    # addresses = getAddress(url, [(paramsold, rnnModelold), (params, rnnModel), (paramslstm, lstmmodel)])
    # titles = getTitle(url, addresses)
    # images = getImg(url)
    # str_to_return = consolidateStuff(url, titles, addresses, images)
