import tensorflow as tf
import pandas as pd
import numpy as np
import HandWritten_cnn
BATCH_SIZE = 100
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 2500,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/minist',
                           """Directory where to write event logs """
                           """and checkpoint.""")

def train():
    train_datas, train_labels, validation_datas, validation_labels = HandWritten_cnn.split_data()
    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_datas.shape[0]
    with tf.Graph().as_default():
        x = tf.placeholder('float', shape=[None, 784])
        y_ = tf.placeholder('float', shape=[None, 10])

        model_labels,regularizers = HandWritten_cnn.inference(x)

        loss = HandWritten_cnn.loss(model_labels,y_,regularizers)

        optimizer = HandWritten_cnn.train(loss)

        # evaluation
        correct_prediction = tf.equal(tf.argmax(model_labels, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

        tf.scalar_summary("loss",loss)
        tf.scalar_summary("accuracy",accuracy)



        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        merge = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/validation')

    for step in range(FLAGS.max_steps):
        batch_x, batch_y, index_in_epoch, epochs_completed = HandWritten_cnn.get_batch(BATCH_SIZE, train_datas, train_labels, index_in_epoch, epochs_completed,num_examples)
        sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y})
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            validation_writer.add_summary(sess.run(merge, feed_dict={x:validation_datas[0:BATCH_SIZE], y_: validation_labels[0:BATCH_SIZE]}), step)
            train_writer.add_summary(sess.run(merge,feed_dict={x: batch_x, y_: batch_y}), step)
        if step ==  (FLAGS.max_steps - 1):
            print ("done")






def main(argv=None):
    train()



if __name__ == '__main__':
  tf.app.run()