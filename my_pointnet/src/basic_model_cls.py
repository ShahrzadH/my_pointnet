'''
Created on Dec 4, 2018

@author: m
'''

from utils import *

def get_model(point_cloud, is_training, bn_decay=None):
    """ Regression PointNet, input is BxNxF, output Bx1 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    num_channel = point_cloud.get_shape()[2].value
    input_image = tf.expand_dims(point_cloud, -1)
    
    is_training = tf.placeholder_with_default(is_training, shape=())
    
    # Point functions (MLP implemented as conv2d)
    net = conv2d(input_image, 64, [1,num_channel],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='conv1', bn_decay=bn_decay)
    net = conv2d(net, 64, [1,1],
                 padding='VALID', stride=[1,1],
                 bn=True, is_training=is_training,
                 scope='conv2', bn_decay=bn_decay)
    net = conv2d(net, 64, [1,1],
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
    
    # MLP on global point cloud vector
    net = tf.reshape(net, [batch_size, -1])
    net = fully_connected(net, 512, bn=True, is_training=is_training,
                          scope='fc1', bn_decay=bn_decay)
    net = fully_connected(net, 256, bn=True, is_training=is_training,
                          scope='fc2', bn_decay=bn_decay)
    net = dropout(net, keep_prob=0.7, is_training=is_training,
                  scope='dp1')
    net = fully_connected(net, 1, activation_fn=None, scope='fc3')

    return net


def get_loss(pred, label):
    """ 
    pred: B,
    label: B, 
    """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label)
    sigmoid_loss = tf.reduce_mean(loss)
    return sigmoid_loss

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
    assert points.get_shape()[-1] == num_channel
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[-1] == 1

    return points, labels



