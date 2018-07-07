from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

import data_iterator

random.seed(0)

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def batch_norm_wrapper(inputs, is_training, decay = 0.999,epsilon=1e-3):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    def f1():
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    def f2():
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)
    return tf.cond(is_training,f1,f2)

class CnnModel(object):
    
    def __init__(self,
                 size,
                 from_vocab_size,
                 target_vocab_size,
                 num_layers,
                 batch_size,
                 max_len,
                 learning_rate,
                 learning_rate_decay_factor,
                 withAdagrad = True,
                 forward_only=False,
                 dropoutRate = 1.0,
                 devices = "",
                 run_options = None,
                 run_metadata = None,
                 dtype=tf.float32,
                 with_attention = False,
                 word_vector = False,
                 dilation_rate=1,
                 ):
        """Create the model.
        
        Args:
        size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.

        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.

        forward_only: if set, we do not construct the backward pass in the model.
        dtype: the data type to use to store internal variables.
        word_vector: use pre-trained word embedding or not.
        """
        self.PAD_ID = 0
        self.UNK_ID = 1
        self.batch_size = batch_size
        self.max_len = max_len
        self.devices = devices
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.dtype = dtype
        self.from_vocab_size = from_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_layers = num_layers
        self.size = size
        self.with_attention = with_attention
        self.word_vector = word_vector
        self.dilation_rate=dilation_rate
        self.dropoutRate=dropoutRate

        # some parameters
        with tf.device(devices[0]):
            self.dropoutRate_place=tf.placeholder(tf.float32,shape=[])
            self.is_training=tf.placeholder(tf.bool,shape=[])
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
            #will this operation be excuted ?doubt it really
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)


        # Input Layer
            self.input_embedding = tf.get_variable("input_embedding",[from_vocab_size, size], dtype = dtype)
            if self.word_vector:
                self.word_embedding_init = tf.placeholder(tf.float32,[from_vocab_size, size], name="word_embedding_init")
                self.input_embedding.assign(self.word_embedding_init)
            self.input_plhd = tf.placeholder(tf.int32, shape = [self.batch_size, self.max_len], name = "input")
            self.input_embed = tf.nn.embedding_lookup(self.input_embedding, self.input_plhd)
            self.input_lens = tf.placeholder(tf.int32, shape = [self.batch_size], name="input_length")

        # CNN layers
            with tf.name_scope('cnn'):
                self.W_conv1=weight_variable([3,self.size,self.size])
                self.b_conv1=bias_variable([self.size])
                self.z_conv1=self.b_conv1+tf.nn.conv1d(self.input_embed,self.W_conv1,1,'SAME',data_format='NWC')
                self.bn_conv1=batch_norm_wrapper(self.z_conv1,self.is_training)
                self.h_conv1=tf.nn.dropout(tf.nn.relu(self.bn_conv1),keep_prob=self.dropoutRate_place)

                self.W_conv2=weight_variable([3,self.size,self.size])
                self.b_conv2=bias_variable([self.size])
                self.z_conv2=self.b_conv2+tf.nn.convolution(self.h_conv1,self.W_conv2,padding='SAME',strides=[1,],dilation_rate=[self.dilation_rate,],data_format='NWC')
                self.bn_conv2=batch_norm_wrapper(self.z_conv2,self.is_training)
                self.h_conv2=tf.nn.dropout(tf.nn.relu(self.bn_conv2),keep_prob=self.dropoutRate_place)

                self.W_conv3=weight_variable([3,self.size,self.size])
                self.b_conv3=bias_variable([self.size])
                self.z_conv3=self.b_conv3+tf.nn.convolution(self.h_conv2,self.W_conv3,padding='SAME',strides=[1,],dilation_rate=[self.dilation_rate,],data_format='NWC')
                self.bn_conv3=batch_norm_wrapper(self.z_conv3,self.is_training)
                self.h_conv3=tf.nn.dropout(tf.nn.relu(self.bn_conv3),keep_prob=self.dropoutRate_place)

                self.W_conv4=weight_variable([3,self.size,self.size])
                self.b_conv4=bias_variable([self.size])
                self.z_conv4=self.b_conv4+tf.nn.conv1d(self.h_conv3,self.W_conv4,1,'SAME',data_format='NWC')
                self.bn_conv4=batch_norm_wrapper(self.z_conv4,self.is_training)
                self.h_conv4=tf.nn.relu(self.bn_conv4)    

                self.hts_flat = tf.reshape(self.h_conv4, [-1, self.size])

        # Output Layer
            self.target = tf.placeholder(tf.int32, shape = [self.batch_size, self.max_len], name = "target")
            self.output_embedding = tf.get_variable("output_embedding",[target_vocab_size, size], dtype = dtype)
            logits = tf.matmul(self.hts_flat, tf.transpose(self.output_embedding))
            self.unary_scores = tf.reshape(logits, [self.batch_size, self.max_len, self.target_vocab_size])
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores, self.target, self.input_lens)
            self.loss = tf.reduce_mean(-log_likelihood)
            self.transition_matrix = transition_params


        # train
            params = tf.trainable_variables()
            if not forward_only:
                if withAdagrad:
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                else:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)

                gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
                # clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                # self.gradient_norms = norm
                self.updates = opt.apply_gradients(zip(gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())
        self.best_saver = tf.train.Saver(tf.global_variables())


    def step(self,session, inputs, targets, lengths, 
        word_embedding = None, forward_only = False):

        input_feed = {}
        input_feed[self.dropoutRate_place]= 1.0 if forward_only else self.dropoutRate
        input_feed[self.is_training]= False if forward_only else True
        input_feed[self.input_plhd.name] = inputs
        input_feed[self.input_lens] = lengths
        input_feed[self.target] = targets
        if word_embedding is not None:
            input_feed[self.word_embedding_init.name] = word_embedding

        # output_feed
        if forward_only:
            output_feed = [self.loss]
            output_feed += [self.unary_scores]
            output_feed += [self.transition_matrix]
        else:
            output_feed = [self.loss]
            output_feed += [self.unary_scores]
            output_feed += [self.transition_matrix]
            #output_feed += [self.updates, self.gradient_norms]
            output_feed += [self.updates]

        outputs = session.run(output_feed, input_feed, options = self.run_options, run_metadata = self.run_metadata)

        return outputs[0], outputs[1], outputs[2]


    def get_batch(self, data_set, start_id = None):

        source_input_ids, target_output_ids, lengths = [], [], []

        for i in xrange(self.batch_size):
            if start_id == None:
                source_seq, target_seq = random.choice(data_set)
            else:
                if start_id + i < len(data_set):
                    source_seq, target_seq = data_set[start_id + i]
                else:
                    source_seq, target_seq = [],[]
           
            lengths.append(len(source_seq))
            source_seq = source_seq + [self.PAD_ID] * (self.max_len - len(source_seq))
            target_seq = target_seq + [self.PAD_ID] * (self.max_len - len(target_seq))

            source_input_ids.append(source_seq)
            target_output_ids.append(target_seq)
        
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set):
            finished = True


        return source_input_ids, target_output_ids, lengths, finished


    def get_test_batch(self, data_set, start_id):

        source_input_ids, lengths = [], []

        for i in xrange(self.batch_size):
            if start_id + i < len(data_set):
                source_seq= data_set[start_id + i]
            else:
                source_seq= []

            #weights.append(len(source_seq))
            
            source_seq =  source_seq + [self.PAD_ID] * (self.max_len - len(source_seq))            
            source_input_ids.append(source_seq)
            lengths.append(len(source_seq))
                    
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set):
            finished = True

        return source_input_ids, lengths, finished
