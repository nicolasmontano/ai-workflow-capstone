from flask import Flask, jsonify, request, send_from_directory
import joblib
import socket
import pandas as pd
import os
import re
from model import model_train, model_predict

app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/predict', methods=['GET','POST'])
def predict():
    
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR API (predict): received request, but no 'query' found within")
        return jsonify([])
    
    query = request.json['query']
    
    y_pred = model_predict(query)
    
    return(jsonify(y_pred.tolist()))

@app.route('/train', methods=['GET','POST'])
def train():
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True
    query = request.json['query']
    print("... training model")
    model_train(data_dir=query,test=test)
    print("... training complete")

    return(jsonify(True))

@app.route('/logs/<filename>',methods=['GET'])
def logs(filename):
    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir,filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])

    return send_from_directory(log_dir, filename, as_attachment=True)        
            
if __name__ == '__main__':
    app.run(port=8080,debug=False)