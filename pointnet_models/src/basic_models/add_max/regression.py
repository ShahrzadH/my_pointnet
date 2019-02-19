'''
Created on Dec 4, 2018

@author: m
'''

from src.utils.tf_utils import *
import tensorflow as tf


class Model(object):
    def __init__(self, is_training, batch_size, num_points, num_channels, model_size, output_channels, \
                 lrn_rate=1e-3, bn_decay=None, optimizer='adam', wd_coef=1e-3):
        self._is_training = is_training
        self._bs = batch_size
        self._np = num_points
        self._nc = num_channels
        self._ms = model_size
        self._oc = output_channels
        self._lrn_rate = lrn_rate
        self._bn_decay = bn_decay
        self._optimizer = optimizer
        self._wd_coef = wd_coef
        
        self.inputs = tf.placeholder(tf.float32, shape=[self._bs, self._np, self._nc], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[self._bs, self._oc], name='labels')
        self.training = tf.placeholder_with_default(self._is_training, shape=())
        
    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        
        if self._is_training:
            self._build_train_op()
            
        self.summaries = tf.summary.merge_all()
        
    def _build_model(self):
        """ Basic Addition Regression Network, input is BxNxC, output Bx1 """
    
        inputs = tf.expand_dims(self.inputs, -1)
        
        # Point functions (MLP implemented as conv2d)
        net = conv2d(inputs, self._ms, [1, self._nc],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=self.training,
                     scope='conv1', bn_decay=self._bn_decay)
        net = conv2d(net, self._ms, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=self.training,
                     scope='conv2', bn_decay=self._bn_decay)
        net = conv2d(net, self._ms, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=self.training,
                     scope='conv3', bn_decay=self._bn_decay)
        net = conv2d(net, self._ms * 2, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=self.training,
                     scope='conv4', bn_decay=self._bn_decay)
        net = conv2d(net, self._ms * 16, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=self.training,
                     scope='conv5', bn_decay=self._bn_decay, activation_fn=None)
    
        # Symmetric function: addition and maximum
        v1 = tf.reduce_sum(net, 1)
        v1 = tf.reshape(v1, [self._bs, -1])
        
        v2 = max_pool2d(net, [self._np, 1],
                        padding='VALID', scope='maxpool')
        v2 = tf.reshape(v2, [self._bs, -1])
        
        net = tf.concat([v1, v2], 1)
        
        # MLP on global point cloud vector
        net = fully_connected(net, self._ms * 8, bn=True, is_training=self.training,
                              scope='fc1', bn_decay=self._bn_decay)
        net = fully_connected(net, self._ms * 4, bn=True, is_training=self.training,
                              scope='fc2', bn_decay=self._bn_decay)
        net = dropout(net, keep_prob=0.7, is_training=self.training,
                      scope='dp1')
        net = fully_connected(net, self._oc, activation_fn=None, scope='fc3')
    
        self.predictions = net  # Record prediction
        
        with tf.variable_scope('losses'):
            reg_loss = tf.reduce_mean(tf.nn.mean_squared_error(predictions=self.predictions, labels=self.labels))
            wd_loss = tf.add_n(tf.get_collection('wd_losses')) 
            
            self.loss = reg_loss + self._wd_coef * wd_loss
            
            tf.summary.scalar('loss', self.loss)
        
    def _build_train_op(self):
        tf.summary.scalar('learning_rate', self._lrn_rate)
        
        if self._optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lrn_rate)
        elif self._optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self._lrn_rate, 0.9)
        else: # Adam
            optimizer = tf.train.AdamOptimizer(self._lrn_rate)
            
        grads, v = zip(*optimizer.compute_gradients(self.loss))
        apply_op = optimizer.apply_gradients(zip(grads, v), \
                                             global_step=self.global_step, name='train_step')

        self.train_op = apply_op
