from flask import Flask
from flask import request
from flask import render_template
from address import parsepage
from title import getTitle
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("form1.html")

@app.route('/', methods=['POST'])
def post_form():
    url = request.form['text']
    addresses = parsepage(url)
    titles = getTitle(url)
    str_to_return = 'Titles:<br>'

    for title in titles:
        str_to_return+=title
        str_to_return+='<br>'

    str_to_return+='<br> Addresses <br>'
    for addr in addresses:
        for part in addr:
            str_to_return+=','.join(part)
            str_to_return+='<br>'

    return str_to_return

if __name__ == '__main__':
    app.run()
