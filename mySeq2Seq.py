import math
import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
from tensorflow.python.layers import core as layers_core
import utils
from nltk.translate.bleu_score import sentence_bleu

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
                 beam_search = False,
                 beam_width = None,
                 mode = None):
        
        self.bidirectional = bidirectional
        self.attention = attention
        self.beam_search = beam_search
        self.mode = mode

        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.encoder_num_units = encoder_num_units
        self.decoder_num_units = decoder_num_units

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.beam_width = beam_width

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

        if self.mode == "Train":
            self._init_optimizer()

    def _init_placeholders(self):
        self.encoder_inputs = tf.placeholder(
            shape = (None, None),
            dtype = tf.int32,
            name = 'encoder_inputs'
        )

        self.encoder_inputs_length = tf.placeholder(
            shape = (None,),
            dtype = tf.int32,
            name='encoder_inputs_length',
        )
        
        self.decoder_targets = tf.placeholder(
            shape = (None, None),
            dtype = tf.int32,
            name = 'decoder_targets',
        )
        
        self.decoder_targets_length = tf.placeholder(
            shape = (None,),
            dtype = tf.int32,
            name = 'decoder_targets_length',
        )
        
    def make_train_inputs(self, x, y):
        inputs, num_inputs = utils.prepare_batch(x)
        targets, num_targets = utils.prepare_batch(y)

        return {
            self.encoder_inputs: inputs,
            self.encoder_inputs_length: num_inputs,
            self.decoder_targets: targets,
            self.decoder_targets_length: num_targets
        }
    
    def make_infer_inputs(self, x):
        inputs, num_inputs = utils.prepare_batch(x)
        
        return{
            self.encoder_inputs: inputs,
            self.encoder_inputs_length: num_inputs
        }
            
    def _init_decoder_train_connectors(self):
        with tf.name_scope('DecoderTrainFeeds'):                
            self.decoder_train_length = self.decoder_targets_length
            self.loss_weights = tf.ones(
                [self.batch_size, tf.reduce_max(self.decoder_train_length)], 
                dtype=tf.float32)

    def _init_embedding(self):
        self.embedding_encoder = tf.Variable(tf.random_uniform(
            [self.vocab_size, 
             self.embedding_size]))
        self.encoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_encoder, 
            self.encoder_inputs)
        
        self.embedding_decoder = tf.Variable(tf.random_uniform(
            [self.vocab_size, 
             self.embedding_size]))
        self.decoder_embedding_inputs = tf.nn.embedding_lookup(
            self.embedding_decoder, 
            self.decoder_targets)
    
    def _init_encoder(self):
        def make_cell(rnn_size):
            enc_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            return enc_cell
        
        with tf.variable_scope("Encoder") as scope:
            num_layers = self.num_layers
            encoder_cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.encoder_num_units) for _ in range(num_layers)])
            
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell = encoder_cell, 
                inputs = self.encoder_embedding_inputs,
                sequence_length = self.encoder_inputs_length, 
                dtype = tf.float32
            )

    def _init_bidirectional_encoder(self):
        ''' to be fixed '''
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
        
        def create_decoder_cell():
            cell = tf.contrib.rnn.MultiRNNCell([make_cell(self.decoder_num_units) for _ in range(self.num_layers)])
                
            if self.beam_search and self.mode == "Infer":
                dec_start_state = seq2seq.tile_batch(self.encoder_state, self.beam_width)
                enc_outputs = seq2seq.tile_batch(self.encoder_outputs, self.beam_width)
                enc_lengths = seq2seq.tile_batch(self.encoder_inputs_length, self.beam_width)
            else:
                dec_start_state = self.encoder_state
                enc_outputs = self.encoder_outputs
                enc_lengths = self.encoder_inputs_length

            if self.attention:
                attention_states = enc_outputs

                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    self.decoder_num_units, 
                    attention_states,
                    memory_sequence_length = enc_lengths)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell,
                    attention_mechanism,
                    attention_layer_size = self.decoder_num_units)

                if self.beam_search and self.mode == "Infer":
                    initial_state = decoder_cell.zero_state(self.batch_size * self.beam_width, tf.float32)
                else:
                    initial_state = decoder_cell.zero_state(self.batch_size, tf.float32)

                initial_state = initial_state.clone(cell_state = dec_start_state)
            else:
                initial_state = dec_start_state
                
            return decoder_cell, initial_state
        
        with tf.variable_scope("Decoder") as scope:
            projection_layer = layers_core.Dense(units = self.vocab_size, use_bias = False) # use_bias
            self.encoder_state = tuple(self.encoder_state[-1] for _ in range(self.num_layers))

            decoder_cell, initial_state = create_decoder_cell()

            if self.mode == "Train":
                    training_helper = tf.contrib.seq2seq.TrainingHelper(
                        self.decoder_embedding_inputs, 
                        self.decoder_train_length)

                    training_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell = decoder_cell, 
                        helper = training_helper, 
                        initial_state = initial_state,
                        output_layer = projection_layer)

                    (self.decoder_outputs_train,
                    self.decoder_state_train,
                    final_sequence_length) = tf.contrib.seq2seq.dynamic_decode(
                            decoder = training_decoder, 
                            impute_finished = True,
                            scope = scope
                    )

                    self.decoder_logits_train = self.decoder_outputs_train.rnn_output
                    decoder_predictions_train = tf.argmax(self.decoder_logits_train, axis = -1)
                    self.decoder_predictions_train = tf.identity(decoder_predictions_train)

            elif self.mode == "Infer":
                    start_tokens = tf.tile(tf.constant([0], dtype=tf.int32), [self.batch_size])

                    if self.beam_search == True:
                        inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                cell          = decoder_cell,
                                embedding     = self.embedding_decoder,
                                start_tokens  = tf.ones_like(self.encoder_inputs_length) * tf.constant(0, dtype = tf.int32),
                                end_token     = tf.constant(1, dtype = tf.int32),
                                initial_state = initial_state,
                                beam_width    = self.beam_width,
                                output_layer  = projection_layer)

                        self.decoder_outputs_inference, __, ___ = tf.contrib.seq2seq.dynamic_decode(
                            decoder = inference_decoder,
                            maximum_iterations = tf.round(tf.reduce_max(self.encoder_inputs_length)) * 2,
                            impute_finished = False,
                            scope = scope)

                        self.decoder_predictions_inference = tf.identity(self.decoder_outputs_inference.predicted_ids)
                    else:
                        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                            self.embedding_decoder,
                            start_tokens = start_tokens, 
                            end_token = 1) # EOS id

                        inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                            cell = decoder_cell, 
                            helper = inference_helper, 
                            initial_state = initial_state,
                            output_layer = projection_layer)

                        self.decoder_outputs_inference, __, ___ = tf.contrib.seq2seq.dynamic_decode(
                            decoder = inference_decoder,
                            maximum_iterations = tf.round(tf.reduce_max(self.encoder_inputs_length)) * 2,
                            impute_finished = False,
                            scope = scope)

                        self.decoder_predictions_inference = tf.identity(self.decoder_outputs_inference.sample_id)

    def _init_optimizer(self):
        loss_mask = tf.sequence_mask(
            tf.to_int32(self.decoder_targets_length), 
            tf.reduce_max(self.decoder_targets_length),
            dtype = tf.float32)
        
        self.loss = tf.contrib.seq2seq.sequence_loss(
            self.decoder_logits_train,
            self.decoder_targets,
            loss_mask)
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()
        
        learning_rate = 0.0002
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = optimizer.apply_gradients(capped_gradients)