# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 13:37:21 2022

@author: user
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# lire les fichiers data_train et data_test
train_ds = pd.read_csv("data_train.txt", delimiter=';')
test_ds = pd.read_csv("data_test.txt", delimiter = ';')

print(train_ds.shape) #afficher les dimensions des  datasets
print(test_ds.shape)
print(test_ds.text)

print(train_ds.head()) # afficher les 5 premiers élements  
print(test_ds.head())

# remplacer -1 et 1 par 0 et 1
training_leb = []
for l in train_ds.label:
    i = 1;
    if l == -1:
        i=0
    training_leb.append(i)
train_ds.drop(columns = ['label'], inplace = True)
train_ds['label'] = training_leb

print(train_ds.head())
print(test_ds.head())


# cleaning data
import re
import string
import unicodedata

def lower_text(text):
    return text.lower() # tout le texte en minuscule

def remove_emails(x):
     return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x) # enlever les @


def remove_urls(x):
    return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)
    # enlever les liens


def remove_rt(x):
    return re.sub(r'\brt\b', '', x).strip()

    
def remove_special_chars(x):
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x
    # enlever les caractères spéciaux 

def remove_accented_chars(x):
    x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return x
    # enlever les accents 

print(train_ds.head())
print(test_ds.head())

from itertools import groupby
# divise les phrases en des mots et les mets dans une liste
def reshape_words(text):
    words = text.split()
    for word in words:
        i = words.index(word)
        chars = [ch for ch in words[i]]
        chars = [x[0] for x in groupby(chars)]
        words[i] = "".join(chars)
    return " ".join(words)


train_ds["text"] = train_ds.text.map(reshape_words)
test_ds["text"] = test_ds.text.map(reshape_words)

test_ds.head()

from collections import Counter

# Count unique words
def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count

# Count unique words
def counter_word_from_list(text_col):
    count = Counter()
    for text in text_col:
        for word in text.split():
            count[word] += 1
    return count

# remplacer les chiffres par leur équivalent en lettres
def remove_special(text):
  filtered_words = [word for word in text.split()]
  for i in range(len(filtered_words)):
    chars = [char for char in filtered_words[i]]
    for j in range(len(chars)):
      if chars[j] == "1" or chars[j] == "2":
        chars[j] = "a"
      elif chars[j] == "4":
        chars[j] = "dh"
      elif chars[j] == "5":
        chars[j] = "kh"
      elif chars[j] == "7":
        chars[j] = "h"
      elif chars[j] == "8":
        chars[j] = "gh"
      elif chars[j] == "9":
        chars[j] = "k"
    filtered_words[i] = "".join(chars)
  return " ".join(filtered_words)


train_ds["text"] = train_ds.text.map(remove_special)
test_ds["text"] = test_ds.text.map(remove_special)

# regrouper les mots qui ont le meme sens mais écrits différemment
def regroup_words(text):
    text = text.lower()
    filtered_words = [word for word in text.split()]
    for i in range(len(filtered_words)):
      if filtered_words[i] == "ou" or filtered_words[i] == "we" or filtered_words[i] == "wa" or filtered_words[i] == "wou":
        filtered_words[i] = "w"
      elif filtered_words[i] == "mn" or filtered_words[i] == "min":
        filtered_words[i] = "men"
      elif filtered_words[i] == "eni" or filtered_words[i] == "ena" or filtered_words[i] == "ani":
        filtered_words[i] = "ana"
      elif filtered_words[i] == "elli" or filtered_words[i] == "illi" or filtered_words[i] == "ili" or filtered_words[i] == "l" or filtered_words[i] == "le" or filtered_words[i] == "il" or filtered_words[i] == "el":
        filtered_words[i] = "li"
      elif filtered_words[i] == "alah":
        filtered_words[i] = "allah"
      elif filtered_words[i] == "m3k":
        filtered_words[i] = "m3ak"
      elif filtered_words[i] == "f" or filtered_words[i] == "fe" or filtered_words[i] == "fel" or filtered_words[i] == "fil":
        filtered_words[i] = "fi"
      elif filtered_words[i] == "bil" or filtered_words[i] == "b" or filtered_words[i] == "be":
        filtered_words[i] = "bel"
      elif filtered_words[i] == "lek":
        filtered_words[i] = "lik"
      elif filtered_words[i] == "asba" or filtered_words[i] == "3sba":
        filtered_words[i] = "3asba"
    return " ".join(filtered_words)

train_ds["text"] = train_ds.text.map(regroup_words)
test_ds["text"] = test_ds.text.map(regroup_words)

counter = counter_word(train_ds.text)
train_ds.head()


#retirer tous les mots qui n’apportent pas vraiment de valeurs à l’analyse globale du texte (mots en francais)
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
frstop = stopwords.words('french')

def remove_stopwords(text):
    text = text.lower()
    filtered_words = [word for word in text.split() if word not in frstop]
    return " ".join(filtered_words)

def lower_text(text):
    return text.lower()

train_ds["text"] = train_ds.text.map(remove_stopwords)
test_ds["text"] = test_ds.text.map(remove_stopwords)

print(train_ds.head())
print(test_ds.head())

def remove_useless(text):
    filtered_words = [word for word in text.split() if len(word)>3 or word == "nik" or word == "si" or word == "ok" or word == "het"]
    return " ".join(filtered_words)

train_ds["text"] = train_ds.text.map(remove_useless)
test_ds["text"] = test_ds.text.map(remove_useless)

all_sentences = list(train_ds.text) + list(test_ds.text)
counter = counter_word_from_list(all_sentences)
counter.most_common()

print(len(counter))

num_unique_words = len(counter)

train_ds = train_ds.sample(frac=1).reset_index(drop=True)
cut = 60000
test_reviews = train_ds.text.to_numpy()
test_labels = train_ds.label.to_numpy()
train_reviews = train_ds.text.to_numpy()
train_labels = train_ds.label.to_numpy()

train_reviews.shape, test_reviews.shape
# définition des paramètres liés au padding
trunc_type='post'
oov_tok = "<OOV>"

# étape de tokenization 
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=num_unique_words, oov_token=oov_tok)
tokenizer.fit_on_texts(train_reviews)

word_index = tokenizer.word_index

# sequencing
train_sequences = tokenizer.texts_to_sequences(train_reviews)
print(train_sequences)

print(train_reviews[10:15])
print(train_sequences[10:15])

# padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 30

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")

print(train_padded.shape)

# définition du modèle 
model = keras.models.Sequential()

model.add(layers.Embedding(num_unique_words, 64, input_length=max_length))

model.add(layers.Bidirectional(layers.LSTM(64)))

model.add(layers.Dense(64, activation="relu"))

model.add(layers.Dense(1,activation="sigmoid"))

model.summary()

# définition de la fonction loss, optimizer et metrics
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# sequencing et padding des données de test
def sample_data():
    test_sequences = tokenizer.texts_to_sequences(test_reviews)
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")
    return test_padded, test_labels

# entrainement du modèle
history = model.fit(train_padded,train_labels,epochs=5,validation_data=(sample_data()))


import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

model.save("sentiment.h5")
import pickle
with open('tokenizer.pickle', 'wb') as handle:
     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

