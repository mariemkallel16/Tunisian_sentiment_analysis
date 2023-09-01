import os
import re
from mimetypes import init
from typing import Any

import numpy as np
import tensorflow as tf

import unicodedata
from flask import Flask, render_template, request
from keras.models import load_model
from nltk import probability
from numpy import array
from tensorboard.plugins import graph
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.backend import clear_session
import pandas as pd

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def init():
    global model, graph
    graph = tf.Graph()


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")


@app.route('/sentiment_prediction', methods=['POST', "GET"])
def sent_anly_prediction():
    if request.method == 'POST':
        text = request.form['text']

        tw = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(tw, maxlen=30, padding="post", truncating="post")

        with graph.as_default():
            # load the pre-trained Keras model
            model = load_model('sentiment.h5')

            probability = model.predict(padded)

        if probability < 0.4:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        else:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'happy.png')
    return render_template('index.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.form['upload-file']
        data = pd.read_csv(file)
        data.columns= ['text']
        texts = list(data.text[:])
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=30, padding="post", truncating="post")
        with graph.as_default():
            # load the pre-trained Keras model
            model = load_model('sentiment.h5')
            predictions = [1 if p > 0.4 else -1 for p in model.predict(padded)]

        return render_template('index.html', data=predictions.count(1),data1=predictions.count(-1))


init()
app.run(debug=True)