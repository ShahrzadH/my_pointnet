'''
Created on Dec 4, 2018

@author: m
'''

from transform_nets import input_transform_net, feature_transform_net
from utils import *
import numpy as np
import tensorflow as tf

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNxF, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    num_feature = point_cloud.get_shape()[2].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = conv2d(input_image, 64, [1,num_feature],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='conv1', bn_decay=bn_decay)
    net = conv2d(net, 64, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = conv2d(net_transformed, 64, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='conv3', bn_decay=bn_decay)
    net = conv2d(net, 128, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='conv4', bn_decay=bn_decay)
    net = conv2d(net, 1024, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = max_pool2d(net, [num_point,1],
                     padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, 512, bn=True, is_training=is_training,
                          scope='fc1', bn_decay=bn_decay)
    net = dropout(net, keep_prob=0.7, is_training=is_training,
                  scope='dp1')
    net = fully_connected(net, 256, bn=True, is_training=is_training,
                          scope='fc2', bn_decay=bn_decay)
    net = dropout(net, keep_prob=0.7, is_training=is_training,
                  scope='dp2')
    net = fully_connected(net, 1, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ 
    pred: B,
    label: B, 
    """
    loss = tf.nn.logistic_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight

def build_input(x, y, batch_size, mode):
    
    num_point = x.shape[1]
    num_channel = x.shape[2]
    
    if mode == 'train':
        example_queue = tf.RandomShuffleQueue(capacity=8 * batch_size, \
                                              min_after_dequeue=4 * batch_size, \
                                              dtypes=[tf.float32, tf.float32], \
                                              shapes=[[num_point, num_channel], [1]])
        num_threads = 8
        
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).repeat(150)
    else:
        example_queue = tf.FIFOQueue(3 * batch_size, \
                                     dtypes=[tf.float32, tf.float32], \
                                     shapes=[[num_point, num_channel], [1]])
        num_threads = 1
        
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    iterator = dataset.make_one_shot_iterator()
    point, label = iterator.get_next()

    example_enqueue_op = example_queue.enqueue([point, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * num_threads))

    points, labels = example_queue.dequeue_many(batch_size)

    assert len(points.get_shape()) == 3
    assert points.get_shape()[0] == batch_size
    assert points.get_shape()[-1] == 3
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[-1] == 1

    return points, labels

