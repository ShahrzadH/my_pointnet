'''
Created on Dec 4, 2018

@author: m
'''

from src.utils.tf_utils import *
import tensorflow as tf

def get_basic_comp(point_cloud, is_training, scope, model_size=64, output_size=64, bn_decay=None):
    """ Regression PointNet component, input is BxNxC, output Bx(output_size) """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    num_channel = point_cloud.get_shape()[2].value

    inputs = tf.expand_dims(point_cloud, -1)
    
    # Point functions (MLP implemented as conv2d)
    net = conv2d(inputs, model_size, [1, num_channel],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_conv1'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, model_size, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_conv2'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, model_size, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_conv3'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, model_size * 2, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_conv4'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, model_size * 16, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_conv5'.format(scope), bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = max_pool2d(net, [num_point,1],
                     padding='VALID', scope='{}_maxpool'.format(scope))
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, model_size * 8, bn=True, is_training=is_training,
                          scope='{}_fc1'.format(scope), bn_decay=bn_decay)
    net = fully_connected(net, model_size * 4, bn=True, is_training=is_training,
                          scope='{}_fc2'.format(scope), bn_decay=bn_decay)
    net = dropout(net, keep_prob=0.7, is_training=is_training,
                  scope='{}_dp1'.format(scope))
    net = fully_connected(net, output_size, activation_fn=None, scope='{}_fc3'.format(scope))

    return net

