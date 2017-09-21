import tensorflow as tf
import numpy as np
import random as rn
internal_dim=1
inp=tf.placeholder(name="inp",dtype=tf.float32,shape=(26,1))
out=tf.placeholder(name="out",dtype=tf.float32,shape=(1,26))
prev_state=tf.placeholder(name="prev_state",dtype=tf.float32,shape=(1,internal_dim))
state=tf.placeholder(name="state",dtype=tf.float32,shape=(1,internal_dim))
prev_w=tf.get_variable("prev_w",initializer=tf.random_normal(shape=(internal_dim,internal_dim),mean=0,stddev=1))
in_w=tf.get_variable("in_w",initializer=tf.random_normal(shape=(26,internal_dim),mean=0,stddev=1))
out_w=tf.get_variable("out_w",initializer=tf.random_normal(shape=(internal_dim,26),mean=0,stddev=1))
state=tf.tanh(tf.matmul(tf.transpose(in_w),inp)+tf.matmul(prev_state,prev_w))
output=tf.nn.softmax(tf.matmul(state,out_w))
prev_state=state
loss=tf.reduce_mean(tf.square(output-out))
opti=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
with open("wordsEn.txt") as word_file:
    english_words = list(word.strip().lower() for word in word_file)
