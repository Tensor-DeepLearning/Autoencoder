# -*- coding:utf-8 -*-
"""
An exmple of the paper <Deep Subspace Clustering with Sparsity Prior>

Author:ymthink
E-mail:yinmiaothink@gmail.com

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

class PARTY(object):

    def __init__(self, shape, batch_size, step_num, learning_rate, data_width, data_length, data):
        self.shape = shape
        self.batch_size = batch_size
        self.step_num = step_num
        self.learning_rate = learning_rate
        self.data_width = data_width
        self.data_length = data_length
        self.data = data
        self.data_num = np.shape(data)[0]

        self.C = np.zeros([self.data_num, self.data_num])

        self.sess = tf.Session()
        self._creat_model()

    def _creat_model(self):
        self.x = tf.placeholder(tf.float32, [None, shape[0]], name='x')
        self.Z = tf.placeholder(tf.float32, [self.data_num, shape[len(shape)-1]], name='Z')
        self.c = tf.placeholder(tf.float32, [None, self.data_num], name='c')
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
        x_s = self.sess.run(self.x_, feed_dict={self.x:self.data[:display_num]})

        fig, ax = plt.subplots(2, display_num)
        for fig_i in range(display_num):
            ax[0][fig_i].imshow(np.reshape(self.data[fig_i,:], (self.data_length, self.data_width)))
            ax[0][fig_i].set_xticks([])
            ax[0][fig_i].set_yticks([])

            ax[1][fig_i].imshow(np.reshape(x_s[fig_i], (self.data_length, self.data_width)))
            ax[1][fig_i].set_xticks([])
            ax[1][fig_i].set_yticks([])
        plt.show()

    def _cal_sparse_prior(self):
        iter_num = 1000
        Xs = data.T

        print('Calculating the sparse prior C matrix ...')

        x_i = tf.placeholder(tf.float32, [self.shape[0], 1], name='x_i')
        X = tf.placeholder(tf.float32, [self.shape[0], self.data_num], name='X')
        c_i = tf.Variable(tf.random_normal([self.data_num, 1]), trainable=True, name='c_i')

        loss = tf.reduce_mean(tf.square(x_i - tf.matmul(X, c_i))) + tf.reduce_mean(tf.abs(c_i))
        opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        init = tf.global_variables_initializer()

        for i in range(self.data_num):
            self.sess.run(init)
            xs_i = np.reshape(Xs[:,i], [self.shape[0], 1])

            for step in range(iter_num):
                _, c = self.sess.run([opt,c_i], feed_dict={X:Xs, x_i:xs_i})

            self.C[i,:] = np.reshape(c, self.data_num)
            print('c_%d'%i, 'completed. There are %d'%(self.data_num-1-i), 'left.')
            
        self.sess.close()


    def train(self):
        self._cal_sparse_prior()
        loss = tf.reduce_mean(tf.square(self.x - self.x_)) + tf.reduce_mean(tf.square(self.z - tf.matmul(self.c, self.Z)))
        opt = self._optimizer(loss, None)
        init = tf.global_variables_initializer()

        display_step = int(self.step_num / 20)
        display_num = 10
        
        self.sess.run(init)

        for step in range(self.step_num):
            i = random.randint(0, self.data_num-1)
            xs = np.reshape(self.data[i,:], [1, -1])
            cs = np.reshape(self.C[i,:], [1,-1])
            Zs = self.sess.run([self.z], feed_dict={self.x:self.data})
            _, l = self.sess.run([opt, loss], 
                feed_dict={self.x:xs, self.c:cs, self.Z:Zs[0]})

            if step % display_step == 0:
                print('Step:', '%04d'%(step+1), 'loss =', '{:.9f}'.format(l))

        print('Optimization completed!')
        self._display(display_num)

        self.sess.close()


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    learning_rate = 0.01
    step_num = 20000
    batch_size = 256
    shape = [784, 256, 128]
    data, _=mnist.train.next_batch(5000)

    ae = PARTY(
        shape=shape, 
        batch_size=batch_size, 
        step_num=step_num, 
        learning_rate=learning_rate, 
        data_width=28, 
        data_length=28, 
        data=data
    )
    ae.train()













