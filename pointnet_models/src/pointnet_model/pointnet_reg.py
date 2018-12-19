'''
Created on Dec 4, 2018

@author: m
'''

from src.utils.tf_utils import *
from src.utils.transform_nets import *
import tensorflow as tf

class PointnetRegModel(object):
    def __init__(self, is_training, batch_size, num_points, num_channels, model_size, \
                 lrn_rate=1e-3, bn_decay=None, optimizer='mom', reg_weight=0.001):
        self._is_training = is_training
        self._bs = batch_size
        self._np = num_points
        self._nc = num_channels
        self._ms = model_size
        self._lrn_rate = lrn_rate
        self._bn_decay = bn_decay
        self._optimizer = 'mom'
        self._reg_weight = reg_weight
        
        self.inputs = tf.placeholder(tf.float32, shape=[self._bs, self._np, self._nc], name='inputs')
        self.labels = tf.placeholder(tf.float32, shape=[self._bs, 1], name='labels')
        self.training = tf.placeholder_with_default(self._is_training, shape=())
        
    def build_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self._build_model()
        
        if self._is_training:
            self._build_train_op()
            
        self.summaries = tf.summary.merge_all()
        
    def _build_model(self):
        end_points = {}
        
        transform = input_transform_net(self.inputs, self.training, 'transform_net1', size=self._ms, bn_decay=self._bn_decay)
            
        inputs_transformed = tf.matmul(self.inputs, transform)
        inputs = tf.expand_dims(inputs_transformed, -1)
        
        # Point functions (MLP implemented as conv2d)
        net = conv2d(inputs, self._ms, [1, self._nc],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=self.training,
                     scope='conv1', bn_decay=self._bn_decay)
        net = conv2d(net, self._ms, [1,1],
                     padding='VALID', stride=[1,1],
                     bn=True, is_training=self.training,
                     scope='conv2', bn_decay=self._bn_decay)

        transform = feature_transform_net(net, self.training, 'transform_net2', size=self._ms, bn_decay=self._bn_decay)
        end_points['transform'] = transform
        
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])
        
        net = conv2d(net_transformed, self._ms, [1,1],
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
                     scope='conv5', bn_decay=self._bn_decay)
        
        # Symmetric function: max pooling
        net = max_pool2d(net, [self._np, 1],
                         padding='VALID', scope='maxpool')
        
        # MLP on global point cloud vector
        net = tf.reshape(net, [self._bs, -1])
        net = fully_connected(net, self._ms * 8, bn=True, is_training=self.training,
                              scope='fc1', bn_decay=self._bn_decay)
        net = fully_connected(net, self._ms * 4, bn=True, is_training=self.training,
                              scope='fc2', bn_decay=self._bn_decay)
        net = dropout(net, keep_prob=0.7, is_training=self.training,
                      scope='dp1')
        net = fully_connected(net, 1, activation_fn=None, scope='fc3')
    
        self.predictions = net  # Record prediction
        
        with tf.variable_scope('losses'):
            reg_loss = tf.losses.mean_squared_error(self.labels, self.predictions)
            reg_loss = tf.reduce_mean(reg_loss, name='reg_loss')
            tf.summary.scalar('reg_loss', reg_loss)
            
            # Enforce the transformation as orthogonal matrix
            transform = end_points['transform'] # BxKxK
            K = transform.get_shape()[1].value
            mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
            mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
            mat_diff_loss = tf.nn.l2_loss(mat_diff) 
            tf.summary.scalar('mat_loss', mat_diff_loss)
        
            self.loss = reg_loss + mat_diff_loss * self._reg_weight
            tf.summary.scalar('tot_loss', self.loss)
        
    def _build_train_op(self):
        tf.summary.scalar('learning_rate', self._lrn_rate)
        
        if self._optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lrn_rate)
        elif self._optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self._lrn_rate, 0.9)
            
        grads, v = zip(*optimizer.compute_gradients(self.loss))
        apply_op = optimizer.apply_gradients(zip(grads, v), \
                                             global_step=self.global_step, name='train_step')

        self.train_op = apply_op

