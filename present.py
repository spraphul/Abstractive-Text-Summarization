
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
tf.reset_default_graph()
#import data_processing
#import data_utils
from os import path
import pickle as pkl
import time
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.WARN)
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import os
from tensorflow.python.client import device_lib
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from IPython.display import clear_output, Image, display, HTML
import json
import numpy as np
import pickle as pkl
import pandas as pd


# In[2]:


f = open('vocab_new2', 'rb')
vocab = pkl.load(f)
f.close()


# In[3]:


len(vocab)


# In[4]:


batch_size = 100
max_para_len = 100
max_ttl_len = 15
hidden_num = 100
vocab_size = len(vocab)
emb_dim = 200
lr = 0.005
prob = 0.5
epochs = 20
ckpt_path = 'model'
attention_states = [batch_size, max_para_len, hidden_num]


# In[5]:


WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
VOCAB_SIZE = 20000
UNK = "UNKNOWN_TOKEN"
limit = {
    'max_descriptions' : 1000,
    'min_descriptions' : 0,
    'max_headings' :70 ,
    'min_headings' : 0,
}


# In[6]:


def filter(line, whitelist):
    ##############################################################
    #   Filters out all characters which are not in whitelist    #
    ##############################################################

    return ''.join([ch for ch in line if ch in whitelist])


# In[7]:


def preprocess(string, vocab):
    string = filter(string.lower(), WHITELIST)
    token_string = string.split(' ')
    new_vocab = vocab[:len(vocab)-2]
    word2id = dict((w,i) for i,w in enumerate(new_vocab))
    temp = []
    for j in range(max_para_len):
        if(j>=len(token_string)):
            temp.append(len(new_vocab)+1)
        else:
            if(token_string[j] in new_vocab):
                temp.append(word2id[token_string[j]])
            else:
                temp.append(len(new_vocab))
    return temp  


# In[44]:


# f = open('save/summary_id_new2', 'rb')
# summary = pkl.load(f)
# f.close()

# f = open('save/vocab_new2', 'rb')
# vocab = pkl.load(f)
# f.close()


# In[8]:


model_name= 'Text Summarizer4'
model_dir = 'model/' + model_name
save_dir = os.path.join(model_dir, "save/")
log_dir = os.path.join(model_dir, "log")


# In[9]:


word2id = dict((w,i) for i, w in enumerate(vocab))
id2word = dict((i,w) for i, w in enumerate(vocab))


# In[33]:


# x= np.array(articles)
# y = np.array(summary)
# a = np.zeros([len(y),1])
# for i in range(len(y)):
#     a[i] = len(vocab)-2
# y = np.concatenate((a,y), axis=1)


# In[10]:


inputs = tf.placeholder(tf.int32, (None, max_para_len), 'inputs')
targets = tf.placeholder(tf.int32, (None, None), 'targets')
outputs = tf.placeholder(tf.int32, (None, None), 'outputs')
keep_prob = tf.placeholder(tf.float32)

word_embedding = tf.Variable(tf.random_uniform((vocab_size, emb_dim), -1.0, 1.0), name='enc_embedding')
input_embed = tf.nn.embedding_lookup(word_embedding, inputs)
output_embed = tf.nn.embedding_lookup(word_embedding, outputs)


with tf.variable_scope('encoding') as encoding_scope:
    lstm_enc = tf.contrib.rnn.BasicLSTMCell(hidden_num)
    final_enc = tf.contrib.rnn.DropoutWrapper(lstm_enc, input_keep_prob=prob)
    ((enc_fw, enc_bw),(enc_fw_final_state, enc_bw_final_state)) = (
        tf.nn.bidirectional_dynamic_rnn(cell_fw=final_enc,
                                      cell_bw = final_enc, inputs=input_embed,dtype=tf.float32)
    )
    
    enc_final_state_c = tf.concat((enc_fw_final_state.c, enc_bw_final_state.c), 1)
    enc_final_state_h = tf.concat((enc_fw_final_state.h, enc_bw_final_state.h), 1)
    
    encoder_final_state = LSTMStateTuple(c=enc_final_state_c, h= enc_final_state_h)
    

with tf.variable_scope('attention') as attention_scope:
    attention_states = tf.transpose(encoder_final_state, [1, 0, 2])
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(hidden_num, attention_states)
    

with tf.variable_scope('decoding') as decoding_scope:
    lstm_dec = tf.contrib.rnn.BasicLSTMCell(hidden_num*2)
    final_dec = tf.contrib.rnn.DropoutWrapper(lstm_dec, input_keep_prob=keep_prob)
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(final_dec, attention_mechanism)
    
    dec_outputs,_ = tf.nn.dynamic_rnn(final_dec, inputs= output_embed, initial_state=encoder_final_state)
    
logits = tf.contrib.layers.fully_connected(dec_outputs,num_outputs=vocab_size, activation_fn = None)


with tf.name_scope('optimization'):
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, max_ttl_len]))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


# In[11]:


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(tf.global_variables_initializer())                          # For initializing all the variables
saver = tf.train.Saver()                                             # For Saving the model
summary_writer = tf.summary.FileWriter(log_dir, sess.graph) 


# In[12]:


saver.restore(sess, save_dir)


# In[13]:


def predict_summary(file):
    source_batch = file
    dec_input = np.zeros((len(source_batch), 1)) + word2id['UNKNOWN_TOKEN']


    for i in range(max_ttl_len):
        batch_logits = sess.run(logits,
                     feed_dict = {inputs: source_batch,
                     outputs: dec_input,keep_prob:prob})
        prediction = batch_logits[:,-1].argmax(axis=-1)
        dec_input = np.hstack([dec_input, prediction[:,None]])

    output = []
    for i in range(len(dec_input)):
        temp = []
        for j in range(1, len(dec_input[i])):
            if(id2word[dec_input[i][j]]=='PAD'):
                continue
            if(id2word[dec_input[i][j]]=="UNKNOWN_TOKEN"):
                temp.append('UNK')
            else:
                temp.append(id2word[dec_input[i][j]])

        temp = ' '.join(temp)
        output.append(temp) 
    
    return output


# In[14]:


f = open('input_file.txt','r+')
x_input = f.read()
x_input = preprocess(x_input, vocab)
x_input = np.array(x_input).reshape((1,100))
output = predict_summary(x_input)
output = str(output)


# In[15]:


f = open('output_file.txt','w+')
f.write(output)
f.close()

