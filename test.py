from flask import render_template
from models import getModel
from title import getTitle
from address import getAddress
from utils import consolidateStuff, getImg
from flask import request
from flask import Flask

app = Flask(__name__)

params = {'GRAD_CLIP': 100, 'NAME': 'RNN', 'SEQ_LENGTH': 1, 'NUM_EPOCHS': 15, 'LEARNING_RATE': 0.01,
          'N_HIDDEN': 512, 'PRINT_FREQ': 5, 'NUM_FEATURES': 9, 'BATCH_SIZE': 32, 'NUM_CLUST': 3}

print params
filename = "../RNN32"
rnnModel = getModel(params, filename)


@app.route('/')
def index():
    return render_template("form1.html")


@app.route('/', methods=['POST'])
def post_form():
    url = request.form['text']
    addresses = getAddress(url, [rnnModel], params)
    titles = getTitle(url, addresses)
    images = getImg(url)
    str_to_return = consolidateStuff(url, titles, addresses, images)
    str_to_return = str_to_return.replace('\n', '<br>')
    return str_to_return

if __name__ == '__main__':
    app.run()
