import tensorflow as tf
import pandas as pd
import numpy as np

VALIDATION_SIZE = 2000
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('regular', '0', """if use regularization to prevent overfitting. 0 = not use""")
LEARNING_RATE = 1e-4


def split_data():
    datas = pd.read_csv('train.csv')
    labels_flat = datas[[0]].values.ravel()
    labels_count = np.unique(labels_flat).shape[0]
    images = datas.iloc[:, 1:].values.astype(np.float32)
    labels = one_hot(labels_flat, labels_count)

    # split data into train&corss_validation
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    return train_images, train_labels, validation_images, validation_labels


def inference(datas):
    kernel_size1 = [5, 5, 1, 32]
    kernel_size2 = [5, 5, 32, 64]

    w1 = weight_variable(kernel_size1)
    b1 = bias_variable([32])
    conv1 = tf.nn.relu(conv(tf.reshape(datas,[-1,28,28,1]),w1)+b1)
    hpool1 = max_pool_2x2(conv1)

    w2 = weight_variable(kernel_size2)
    b2 = bias_variable([64])
    conv2 = tf.nn.relu(conv(hpool1,w2)+b2)
    hpool2 = max_pool_2x2(conv2)

    w3 = weight_variable([7*7*64, 1024])
    b3 = bias_variable([1024])
    fc3 = tf.nn.dropout(tf.nn.relu( tf.matmul(tf.reshape(hpool2,[-1,7*7*64]), w3)+b3),0.5)



    w4 = weight_variable([1024, 10])
    b4 = bias_variable([10])



    labels = tf.nn.softmax(tf.matmul(fc3, w4) + b4)

    if FLAGS.regular:
        regularizer = regularizers = (tf.nn.l2_loss(w3)+tf.nn.l2_loss(b3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(b4))
    else:
        regularizer = 0
    return labels, regularizer


def loss(model_labels, labels, regularizers):
    loss = tf.reduce_sum(tf.square(model_labels - labels))
    if regularizers != 0:
        loss += 5e-4 * regularizers
    return loss


def train(loss):
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    return train_step


# convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# ...
# 9 => [0 0 0 0 0 0 0 0 0 1]
def one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def feature_scaling(datas):
    return datas.iloc[:, 1:].values.astype(np.float)


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_batch(batch_size, train_images, train_labels, index_in_epoch, epochs_completed, num_examples):
    start = index_in_epoch
    index_in_epoch += batch_size
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch, epochs_completed


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv(datas, parameter):
    return tf.nn.conv2d(datas, parameter, strides=[1, 1, 1, 1], padding='SAME')
