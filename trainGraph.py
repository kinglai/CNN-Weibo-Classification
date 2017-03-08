"""
Train convolutional network for sentiment analysis. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf



* it turns out that such a small data set as "Movie reviews with one
sentence per review"  (Pang and Lee, 2005) requires much smaller network
than the one introduced in the original article:
- embedding dimension is only 20 (instead of 300; 'CNN-static' still requires ~100)
- 2 filter sizes (instead of 3)
- higher dropout probabilities and
- 3 filters per filter size is enough for 'CNN-non-static' (instead of 100)
- embedding initialization does not require prebuilt Google Word2Vec data.
Training Word2Vec on the same "Movie reviews" data set is enough to 
achieve performance reported in the article (81.6%)

** Another distinct difference is slidind MaxPooling window of length=2
instead of MaxPooling over whole feature map as in the article
"""

import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.utils.np_utils import to_categorical

np.random.seed(2)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for 
# Sentence Classification, Section 3 for detail.

model_variation = 'CNN-rand'  #  CNN-rand | CNN-non-static | CNN-static
print('Model variation is %s' % model_variation)

#test rate 30% lai 2016-12-11
rate=0.3
# Model Hyperparameters
#sequence_length = 56
sequence_length = 58

embedding_dim = 20          
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

# Training parameters
batch_size = 32

#num_epochs = 100

num_epochs=5
val_split = 0.1

# Word2Vec parameters, see train_word2vec
min_word_count = 1  # Minimum word count                        
context = 10        # Context window size    

# Data Preparatopn
# ==================================================
#
# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = data_helpers.load_data()

if model_variation=='CNN-non-static' or model_variation=='CNN-static':
    embedding_weights = train_word2vec(x, vocabulary_inv, embedding_dim, min_word_count, context)
    if model_variation=='CNN-static':
        x = embedding_weights[0][x]
elif model_variation=='CNN-rand':
    embedding_weights = None
else:
    raise ValueError('Unknown model variation')    
lenData=len(y)

y=to_categorical(np.asarray(y))
# Shuffle data
shuffle_indices = np.random.permutation(np.arange(lenData))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

print "x_shuffled shape:" ,x_shuffled.shape
#y_shuffled = y[shuffle_indices].argmax(axis=1)

thre=(int)(lenData*rate)
x_test=[]
x_train=[]
y_test=[]
y_train=[]

for i in range(thre):
    x_test.append(x_shuffled[i])
    y_test.append(y_shuffled[i])

for i in range(thre,lenData):
    x_train.append(x_shuffled[i])
    y_train.append(y_shuffled[i])

x_test=np.array(x_test)
y_test=np.array(y_test)

x_train=np.array(x_train)
y_train=np.array(y_train)

print "x_train shape:",x_train.shape

print("Vocabulary Size: {:d}".format(len(vocabulary)))

# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in parallel
graph_in = Input(shape=(sequence_length, embedding_dim))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)
    
if len(filter_sizes)>1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)

# main sequential model
model = Sequential()
if not model_variation=='CNN-static':
    model.add(Embedding(len(vocabulary), embedding_dim, input_length=sequence_length,
                        weights=embedding_weights))
model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))

#
model.add(Dense(4))
model.add(Activation("softmax"))
#
'''
model.add(Dense(1))
model.add(Activation('sigmoid'))
'''

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Training model
# ==================================================
'''
model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
          nb_epoch=num_epochs, validation_split=val_split, verbose=2)
'''

#################################################
model.fit(x_train, y_train, batch_size=batch_size,
          nb_epoch=num_epochs, validation_split=val_split, verbose=2)

#test
test=model.evaluate(x_test,y_test,batch_size=batch_size,verbose=1)
print test