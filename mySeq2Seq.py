import math
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.python.layers import core as layers_core
import utils

class Seq2SeqModel():
    def __init__(self, 
                 encoder_num_units, 
                 decoder_num_units, 
                 embedding_size,
                 num_layers,
                 vocab_size, 
                 batch_size,
                 bidirectional = False,
                 attention = False,
                 pre_emb = False,
                 time_major = False):
        self.bidirectional = bidirectional
        self.attention = attention
        self.pre_emb = pre_emb
        self.time_major = time_major


        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units

        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self._make_graph()

    def _make_graph(self):

        self._init_placeholders()
        
        self._init_decoder_train_connectors()
        
        self._init_embedding()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_encoder()

        self._init_decoder()

        self._init_optimizer()

    def _init_placeholders(self):
        if self.pre_emb:
            self.encoder_inputs = tf.placeholder(
                shape=(None, None, self.embedding_size),
                dtype=tf.float32,
                name='encoder_inputs',
            )
        else:
            self.encoder_inputs = tf.placeholder(
                shape = (None, None),
                dtype = tf.int32,
                name = 'encoder_inputs'
            )

        self.encoder_inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length',
        )

        if self.pre_emb:
            self.decoder_train_inputs = tf.placeholder(
                shape=(None, None, self.embedding_size),
                dtype=tf.float32,
                name='decoder_train_inputs'
            )
        
        self.decoder_targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_targets',
        )
        
        self.decoder_targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_targets_length',
        )
        
    def make_train_inputs(self, input_seq, input_seq_id, target_seq, target_seq_id):

        inputs_, inputs_length_ = utils.prepare_batch(inputs = input_seq, 
                                                      dim = 3, 
                                                      emb_size = self.embedding_size,
                                                      time_major = self.time_major)
        inputs_id_, inputs_id_length_ = utils.prepare_batch(inputs = input_seq_id, 
                                                            dim = 2,
                                                            time_major = self.time_major)
        targets_, targets_length_ = utils.prepare_batch(inputs = target_seq, 
                                                        dim = 3, 
                                                        emb_size = self.embedding_size,
                                                        time_major = self.time_major)
        targets_id_, targets_id_length_ = utils.prepare_batch(inputs = target_seq_id,
                                                              dim = 2,
                                                              time_major = self.time_major)
        
        if self.pre_emb:
            return {
                self.encoder_inputs: inputs_,
                self.encoder_inputs_length: inputs_length_,
                self.decoder_train_inputs: targets_,
                self.decoder_targets_length: targets_length_,
                self.decoder_targets: targets_id_,
            }
        else:
            return {
                self.encoder_inputs: inputs_id_,
                self.encoder_inputs_length: inputs_id_length_,
                self.decoder_targets_length: targets_id_length_,
                self.decoder_targets: targets_id_,
            }

    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):                
            self.decoder_train_length = self.decoder_targets_length
            self.loss_weights = tf.ones([
                self.batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32)

    def _init_embedding(self):
        if self.pre_emb:
            self.encoder_embedding_inputs = self.encoder_inputs
            self.decoder_embedding_inputs = self.decoder_train_inputs
            
        else:
            embedding_encoder = tf.Variable(tf.random_uniform([self.vocab_size, 
                                                            self.embedding_size]))
            self.encoder_embedding_inputs = tf.nn.embedding_lookup(
                embedding_encoder, self.encoder_inputs)

            embedding_decoder = tf.Variable(tf.random_uniform([self.vocab_size, 
                                                            self.embedding_size]))
            
            self.decoder_embedding_inputs = tf.nn.embedding_lookup(
                embedding_decoder, self.decoder_targets)

    def _init_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            def make_cell(rnn_size):
                enc_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
                return enc_cell
            num_layers = self.num_layers
            encoder_cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.encoder_num_units) for _ in range(num_layers)])
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell = encoder_cell, 
                inputs = self.encoder_embedding_inputs,
                sequence_length = self.encoder_inputs_length, 
                time_major = self.time_major,
                dtype = tf.float32
            )

    def _init_bidirectional_encoder(self):
        '''
            to be fixed
            
        '''
#         with tf.variable_scope("Bidirectional_Encoder") as scope:
#             def make_cell(rnn_size):
#                 enc_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
#                 return enc_cell
#             num_layers = 4
#             encoder_cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.encoder_num_units) for _ in range(num_layers)])
#             bi_enc_outputs, bi_enc_state = tf.nn.bidirectional_dynamic_rnn(
#                 cell_fw = encoder_cell, 
#                 cell_bw = encoder_cell, 
#                 inputs = self.encoder_embedding_inputs,
#                 sequence_length = self.encoder_inputs_length, 
#                 time_major = True,
#                 dtype = tf.float32
#             )
#             self.encoder_outputs = tf.concat(bi_enc_outputs, 2)
            
#             encoder_state_c = tf.concat(
#                 (bi_enc_state[0][0], bi_enc_state[1][0]), -1)
#             encoder_state_h = tf.concat(
#                 (bi_enc_state[0][1], bi_enc_state[1][1]), -1)
#             self.encoder_state = LSTMStateTuple(c = encoder_state_c, h = encoder_state_h)
            
#             encoder_state = []
#             for layer_id in range(num_layers):
#                 encoder_state.append(bi_enc_state[0][layer_id])  # forward
#                 encoder_state.append(bi_enc_state[1][layer_id])  # backward
#             self.encoder_state = tuple(encoder_state)
            
#             self.encoder_state = tf.concat(bi_enc_state, 0)

    def _init_decoder(self):
        def make_cell(rnn_size):
            dec_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            return dec_cell

        num_layers = self.num_layers
        decoder_cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.decoder_num_units) for _ in range(num_layers)])

        projection_layer = layers_core.Dense(units = self.vocab_size, use_bias=False)
    
        if self.attention:
            # attention_states: [batch_size, max_time, num_units]
            if self.time_major == True:
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
            else:
                attention_states = self.encoder_outputs

            # Create an attention mechanism
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.decoder_num_units, 
                attention_states,
                memory_sequence_length = self.encoder_inputs_length)

            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                decoder_cell, 
                attention_mechanism,
                attention_layer_size = self.decoder_num_units)
            
            initial_state = decoder_cell.zero_state(batch_size = self.batch_size, dtype = tf.float32)
            initial_state = initial_state.clone(cell_state = self.encoder_state)
        else:
            initial_state = self.encoder_state

        with tf.variable_scope("Decoder") as scope:
            # Helper
            helper = tf.contrib.seq2seq.TrainingHelper(
                self.decoder_embedding_inputs, 
                self.decoder_train_length, 
                time_major = self.time_major)

            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, 
                helper, 
                initial_state,
                output_layer = projection_layer)

            # Dynamic decoding
            (self.decoder_outputs_train,
            self.decoder_state_train,
            final_sequence_length) = tf.contrib.seq2seq.dynamic_decode(
                    decoder, 
                    scope=scope
            )
            self.decoder_logits_train = self.decoder_outputs_train.rnn_output
            #self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

        with tf.variable_scope("Decoder", reuse = True) as scope:
            embedding_decoder = tf.Variable(tf.random_uniform([self.vocab_size, 
                                                            self.embedding_size]))
            start_tokens = tf.tile(tf.constant([0], dtype=tf.int32), 
                                    [self.batch_size], 
                                    name='start_tokens')
            
            # Helper
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding_decoder,
                start_tokens = start_tokens, 
                end_token = 1) # EOS id

            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                decoder_cell, 
                helper, 
                initial_state,
                output_layer = projection_layer)

            # Dynamic decoding
            self.decoder_outputs_inference, __, ___ = tf.contrib.seq2seq.dynamic_decode(
                decoder = decoder,
                maximum_iterations = tf.round(tf.reduce_max(self.encoder_inputs_length)),
                impute_finished = True)
                # maximum iterations

            self.decoder_logits_inference = tf.identity(self.decoder_outputs_inference.sample_id,
                                                        name = 'predictions')

    def _init_optimizer(self):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels = self.decoder_targets,
            logits = self.decoder_logits_train
        )
        
        # Mask out the losses we don't care about
        
        '''
        TOTRY: tf.sequence_loss
        
        '''
        loss_mask = tf.sequence_mask(
            tf.to_int32(self.decoder_targets_length), 
            tf.reduce_max(self.decoder_targets_length))
        
        if self.time_major == True:
            losses = crossent * tf.transpose(tf.to_float(loss_mask), [1, 0])
        else:
            losses =  tf.to_float(loss_mask) * crossent

        train_loss = tf.reduce_sum(losses) / tf.cast(self.batch_size, tf.float32)
        self.loss = train_loss
        
        # Calculate and clip gradients
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, 
            5 # max_gradient_norm, usually 5 or 1
        )
        
        # Optimization
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))
        self.train_op = optimizer.minimize(self.loss)