'''
Created on Dec 4, 2018

@author: m
'''
import tensorflow as tf
from utils import *
import numpy as np

def input_transform_net(point_cloud, is_training, bn_decay=None):
    """ Input (XYZ) Transform Net, input is BxNxF Point Cloud
        Return:
            Transformation matrix of size FxK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    num_feature = point_cloud.get_shape()[2].value

    input_image = tf.expand_dims(point_cloud, -1)
    net = conv2d(input_image, 64, [1,num_feature],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='tconv1', bn_decay=bn_decay)
    net = conv2d(net, 128, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='tconv2', bn_decay=bn_decay)
    net = conv2d(net, 1024, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='tconv3', bn_decay=bn_decay)
    net = max_pool2d(net, [num_point,1],
                     padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_input') as sc:
        weights = tf.get_variable('weights', [256, num_feature*num_feature],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [num_feature*num_feature],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(num_feature).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, num_feature, num_feature])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    net = conv2d(inputs, 64, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='tconv1', bn_decay=bn_decay)
    net = conv2d(net, 128, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='tconv2', bn_decay=bn_decay)
    net = conv2d(net, 1024, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='tconv3', bn_decay=bn_decay)
    net = max_pool2d(net, [num_point,1],
                     padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, 512, bn=True, is_training=is_training,
                          scope='tfc1', bn_decay=bn_decay)
    net = fully_connected(net, 256, bn=True, is_training=is_training,
                          scope='tfc2', bn_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform