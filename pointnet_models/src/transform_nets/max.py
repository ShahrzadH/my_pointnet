'''
Created on Dec 4, 2018

@author: m
'''
import tensorflow as tf
from src.utils.tf_utils import *
import numpy as np

def input_transform_net(point_cloud, is_training, scope, size=64, bn_decay=None):
    """ Input Transform Net, input is BxNxC Point Cloud
        Return:
            Transformation matrix of size CxC """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    num_channel = point_cloud.get_shape()[2].value

    input_points = tf.expand_dims(point_cloud, -1)
    net = conv2d(input_points, size, [1,num_channel],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_tconv1'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, size * 2, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_tconv2'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, size * 16, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_tconv3'.format(scope), bn_decay=bn_decay)
    net = max_pool2d(net, [num_point,1],
                     padding='VALID', scope='{}_tmaxpool'.format(scope))

    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, size * 8, bn=True, is_training=is_training,
                                  scope='{}_tfc1'.format(scope), bn_decay=bn_decay)
    net = fully_connected(net, size * 4, bn=True, is_training=is_training,
                                  scope='{}_tfc2'.format(scope), bn_decay=bn_decay)

    with tf.variable_scope('{}_transform_input'.format(scope), reuse=tf.AUTO_REUSE) as sc:
        weights = tf.get_variable('weights', [size * 4, num_channel*num_channel],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [num_channel*num_channel],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(num_channel).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, num_channel, num_channel])
    return transform


def feature_transform_net(inputs, is_training, scope, size=64, bn_decay=None):
    """ Feature Transform Net, input is BxNx1xC
        Return:
            Transformation matrix of size CxC """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value
    num_channel = inputs.get_shape()[3].value

    net = conv2d(inputs, size, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_tconv1'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, size * 2, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_tconv2'.format(scope), bn_decay=bn_decay)
    net = conv2d(net, 1024, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='{}_tconv3'.format(scope), bn_decay=bn_decay)
    net = max_pool2d(net, [num_point,1],
                     padding='VALID', scope='{}_tmaxpool'.format(scope))

    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, size * 8, bn=True, is_training=is_training,
                          scope='{}_tfc1'.format(scope), bn_decay=bn_decay)
    net = fully_connected(net, size * 4, bn=True, is_training=is_training,
                          scope='{}_tfc2'.format(scope), bn_decay=bn_decay)

    with tf.variable_scope('{}_transform_feat'.format(scope)) as sc:
        weights = tf.get_variable('weights', [size * 4, num_channel*num_channel],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [num_channel*num_channel],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(num_channel).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, num_channel, num_channel])
    return transform