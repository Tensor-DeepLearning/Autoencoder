# -*- coding:utf-8 -*-
"""
An antoencoder class example.

Author:ymthink
E-mail:yinmiaothink@gmail.com

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Autoencoder(object):

    def __init__(self, shape, batch_size, step_num, learning_rate, data_width, data_length, data):
        self.shape = shape
        self.batch_size = batch_size
        self.step_num = step_num
        self.learning_rate = learning_rate
        self.data_width = data_width
        self.data_length = data_length
        self.data = data

        self._creat_model()

    def _creat_model(self):
        self.x = tf.placeholder(tf.float32, [None, shape[0]], name='x')
        current_input = self.x
        shape_len = len(self.shape)
        with tf.variable_scope('Encoder'):
            for enc_i in range(shape_len-1):
                W = tf.Variable(tf.random_normal([self.shape[enc_i], self.shape[enc_i+1]]), name='W'+str(enc_i))
                b = tf.Variable(tf.zeros([shape[enc_i+1]]), name='b'+str(enc_i))

                current_output = tf.nn.sigmoid(tf.add(tf.matmul(current_input, W), b))
                current_input = current_output

        self.z = current_output

        with tf.variable_scope('Decoder'):
            for dec_i in range(shape_len-1):
                W = tf.Variable(tf.random_normal([self.shape[shape_len-1-dec_i], self.shape[shape_len-2-dec_i]]), name='W'+str(dec_i))
                b = tf.Variable(tf.zeros([self.shape[shape_len-2-dec_i]]), name='b'+str(dec_i))
                current_output = tf.nn.sigmoid(tf.add(tf.matmul(current_input, W), b))
                current_input = current_output
        self.x_ = current_output

    def _optimizer(self, loss, var_list):
        decay = 0.96
        decay_step_num = self.batch_size // 5
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            batch,
            decay_step_num,
            decay,
            staircase=True
        )
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch, var_list=var_list)
        return opt
    
    def _display(self, display_num):
        zs = self.sess.run(self.x_, feed_dict={self.x:data.test.images[:display_num]})

        fig, ax = plt.subplots(2, display_num)
        for fig_i in range(display_num):
            ax[0][fig_i].imshow(np.reshape(data.test.images[fig_i], (self.data_length, self.data_width)))
            ax[0][fig_i].set_xticks([])
            ax[0][fig_i].set_yticks([])

            ax[1][fig_i].imshow(np.reshape(zs[fig_i], (self.data_length, self.data_width)))
            ax[1][fig_i].set_xticks([])
            ax[1][fig_i].set_yticks([])
        plt.show()

    def train(self):
        loss = tf.reduce_mean(tf.pow((self.x - self.x_), 2))
        opt = self._optimizer(loss, None)
        init = tf.global_variables_initializer()
        display_step = 1
        display_num = 10
        
        self.sess = tf.Session()
        self.sess.run(init)

        total_batch = int(self.data.train.num_examples / self.batch_size)
        for step in range(self.step_num):
            for batch_i in range(total_batch):
                batch_xs, batch_ys = self.data.train.next_batch(batch_size)
                _, l = self.sess.run([opt, loss], feed_dict={self.x:batch_xs})

            if step % display_step == 0:
                print('Step:', '%04d'%(step+1), 'loss =', '{:.9f}'.format(l))

        print('Optimization completed!')
        self._display(display_num)

        self.sess.close()


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    data = input_data.read_data_sets("MNIST_data", one_hot=True)

    learning_rate = 0.01
    step_num = 30
    batch_size = 256
    shape = [784, 256, 128]

    ae = Autoencoder(
        shape=shape, 
        batch_size=batch_size, 
        step_num=step_num, 
        learning_rate=learning_rate, 
        data_width=28, 
        data_length=28, 
        data=data
    )
    ae.train()









