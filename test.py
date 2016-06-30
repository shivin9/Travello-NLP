from flask import render_template
# from address import parsepage
from driver import getStuff
from title import getTitle
from flask import request
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("form1.html")

@app.route('/', methods=['POST'])
def post_form():
    url = request.form['text']
    str_to_return = getStuff(url)
    # addresses = parsepage(url)
    # titles = getTitle(url, addresses)
    # str_to_return = 'Titles:<br>'

    # for title in titles:
    #     str_to_return+=title
    #     str_to_return+='<br>'

    # str_to_return+='<br> Addresses <br>'
    # for addr in addresses:
    #     for part in addr:
    #         str_to_return+=','.join(part)
    #         str_to_return+='<br>'

    return str_to_return

if __name__ == '__main__':
    app.run()