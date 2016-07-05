from models import getModel
from title import getTitle
from address import getAddress
from utils import consolidateStuff, getImg
import SocketServer
import os


class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        url = self.data
        print "CLIENT SENT:" + str(url)
        addresses = getAddress(url, [rnnModel], params)
        titles = getTitle(url, addresses)
        images = getImg(url)
        str_to_return = consolidateStuff(url, titles, addresses, images)
        str_to_return = str_to_return.replace('\n', '<br>')

        # send the required data
        self.request.sendall(str_to_return)

if __name__ == "__main__":
    params = {'GRAD_CLIP': 100, 'NAME': 'RNN', 'SEQ_LENGTH': 1, 'NUM_EPOCHS': 15,
              'LEARNING_RATE': 0.01, 'N_HIDDEN': 512, 'PRINT_FREQ': 5, 'NUM_FEATURES': 9,
              'BATCH_SIZE': 32, 'NUM_CLUST': 3}

    filename = "../RNN32"
    rnnModel = getModel(params, filename)
    f = os.popen('ifconfig | grep "inet\ addr" | cut -d: -f2 | cut -d" " -f1')
    IPaddr = f.read().split('\n')[0]

    HOST, PORT = IPaddr, 1728

    # Create the server, binding to localhost on port 1728
    server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
    print "Server is UP!"
    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()
