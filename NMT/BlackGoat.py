import numpy as np
import tensorflow as tf
import json
import re
import gensim
from gensim.models import Word2Vec
import jieba
from Kaneki import Seq2SeqModel

word2vec = Word2Vec.load('../../word2vec/zh_fb_ptt.bin')

concept = []

with open('../../Story/Concept_01.txt', encoding = 'utf8') as c:
    concept = c.readlines()
    
with open('../../Story/Concept_02.txt', encoding = 'utf8') as c:
    concept += c.readlines()
    
story = []
    
with open('../../Story/Story_01.txt', encoding = 'utf8') as c:
    story = c.readlines()
    
with open('../../Story/Story_02.txt', encoding = 'utf8') as c:
    story += c.readlines()
    
for i in range(0, len(concept)):
    concept[i] = concept[i].split(' ')
    concept[i].pop()
    
for i in range(len(story)):
    story[i] = story[i].replace('\n', '')
    
print(len(concept), len(story))

X_train_emb = []
X_train = []

y_train_emb = []
y_train = []

comma_period = ['，', '。']

tokens = ['GO',
          'EOS',
          'unknown',
          '，',
          '。']

relations = ['CapableOf', 
             'Causes', 
             'CausesDesire', 
             'HasFirstSubevent', 
             'HasSubevent', 
             'MotivatedByGoal', 
             'UsedFor']

vocab_list = tokens + relations

ones_init = np.ones(shape = (1, 300))

relations_embedding = []
relations_id = []

tokens_id = []
tokens_embedding = []

x = 5

for i in range(0, x):
    tokens_id += [i]
    tokens_embedding += [ones_init * i / 5]
    
for i in range(x, x + 7):
    relations_id += [i]
    relations_embedding += [ones_init * i / 5]

sigma = 1e-8

def build_vocab_list(vl, w):
    if w in vl:
        return vl, vl.index(w)
    else:
        vl += [w]
        return vl, vl.index(w)

def np_arr_concat(x, y):
    if len(x) == 0:
        return np.array(y)
    else:
        return np.concatenate((x, y), axis = 0)


for a in range(len(concept)):
    word_vector = np.array([])
    id_vector = []
    for i in range(len(concept[a])):
        w = concept[a][i]
        if w in relations:
            i = relations.index(w)
            word_vector = np_arr_concat(word_vector, relations_embedding[i])
            vocab_list, vli = build_vocab_list(vocab_list, w)
            id_vector += [vli]
            
        else:
            try:
                wvec = word2vec[w] + sigma
                word_vector = np_arr_concat(word_vector, [wvec])
                vocab_list, vli = build_vocab_list(vocab_list, w)
                id_vector += [vli]
            except:
                segs = jieba.cut(w)
                for seg in segs:
                    try:
                        wvec = word2vec[seg] + sigma
                        word_vector = np_arr_concat(word_vector, [wvec])
                        vocab_list, vli = build_vocab_list(vocab_list, seg)
                        id_vector += [vli]
                    except:
                        word_vector = np_arr_concat(word_vector, tokens_embedding[2])
                        id_vector += [tokens_id[2]]
                        
    word_vector = np_arr_concat(word_vector, tokens_embedding[1])
    id_vector += [tokens_id[1]]
    X_train_emb += [word_vector]
    X_train += [id_vector]
        
print(len(X_train_emb))
print(len(X_train))
print(len(vocab_list))

for a in range(len(story)):
    # append <GO> in the front
    word_vector = np.array(tokens_embedding[0])
    id_vector = [tokens_id[0]]
    segs = jieba.cut(story[a])
    for seg in segs:
        if seg in comma_period:
            i = comma_period.index(seg)
            word_vector = np_arr_concat(word_vector, tokens_embedding[i + 3])
            vocab_list, vli = build_vocab_list(vocab_list, seg)
            id_vector += [tokens_id[i + 3]]
        else:
            try:
                wvec = word2vec[seg] + sigma
                word_vector = np_arr_concat(word_vector, [wvec])
                vocab_list, vli = build_vocab_list(vocab_list, seg)
                id_vector += [vli]
            except:
                word_vector = np_arr_concat(word_vector, tokens_embedding[2])
                id_vector += [tokens_id[2]]
    word_vector = np_arr_concat(word_vector, tokens_embedding[1])
    id_vector += [tokens_id[1]]
    y_train_emb += [word_vector]
    y_train += [id_vector]

print(len(y_train_emb))
print(len(y_train))
print(len(vocab_list))

import tensorflow as tf

tf.reset_default_graph()
tf.set_random_seed(1)

a = X_train_emb[0:455]
b = X_train[0:455]
c = y_train_emb[0:455]
d = y_train[0:455]

step = 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config = config) as session:

    model = Seq2SeqModel(
        encoder_num_units = 300, 
        decoder_num_units = 300, 
        vocab_size = len(vocab_list), 
        embedding_size = 300,
        num_layers = 1,
        bidirectional = False,
        attention = True,
        pre_emb = True
    )
    
    
    print('model constructed.')
    
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    batch_size = 32
    max_batches = int(len(X_train_emb) / batch_size)
    batches_in_epoch = 50
    
    loss_track = []
    try:
        print('start training.')
        for _batch in range(max_batches + 1):
            X_emb, X, y_emb, y = a, b, c, d
#             X, y, y_id = Touka.input_generator(X_train, y_train, y_train, batch_size)
#             X_emb, X, y_emb, y = tf.train.batch(tensors = [X_train_emb, X_train, y_train_emb, y_train],
#                                                 batch_size = batch_size, 
#                                                 enqueue_many = True,
#                                                 allow_smaller_final_batch = True)
            feed_dict = model.make_train_inputs(X_emb, X, y_emb, y)
            _, l = session.run([model.train_op, model.loss], feed_dict)
            loss_track.append(l)
            
            verbose = True
            if verbose:
                if _batch == 0 or _batch % 33 == 0:
                    print('batch {}'.format(_batch))
                    print('  minibatch loss: {}'.format(session.run(model.loss, feed_dict)))
#                     test_accuracy = tf.contrib.metrics.accuracy(prediction, y_id)
#                     print(test_accuracy)
            if _batch % batches_in_epoch == 0:
                saver.save(session, '../../model/' + 'model.ckpt', global_step = step + 1)
                step = step + 1
                print('model saved at step =', step)
        print('finish training')
    except KeyboardInterrupt:
        print('training interrupted')