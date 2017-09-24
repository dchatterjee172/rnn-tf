import tensorflow as tf
import numpy as np
import random as rn
internal_dim=1
data_len=10
total_data=20000
data=np.zeros(shape=(total_data,data_len))
for i in range(0,total_data):
    for j in range(0,data_len):
        if rn.uniform(0,1)>.5:
            data[i][j]=1
inp=tf.placeholder(name="inp",dtype=tf.float32,shape=(1,1))
out=tf.placeholder(name="out",dtype=tf.float32,shape=(1,1))
prev_state=tf.placeholder(name="prev_state",dtype=tf.float32,shape=(1,internal_dim))
ploss=tf.placeholder(name="ploss",dtype=tf.float32,shape=())
b=tf.get_variable("b",initializer=tf.random_normal(shape=(1,internal_dim),mean=0,stddev=1))
prev_w=tf.get_variable("prev_w",initializer=tf.random_normal(shape=(internal_dim,internal_dim),mean=0,stddev=1))
in_w=tf.get_variable("in_w",initializer=tf.random_normal(shape=(1,internal_dim),mean=0,stddev=1))
out_w=tf.get_variable("out_w",initializer=tf.random_normal(shape=(internal_dim,1),mean=0,stddev=1))
state=tf.nn.elu(tf.matmul(inp,in_w)+tf.matmul(prev_state,prev_w)+b)
output=tf.nn.elu(tf.matmul(state,out_w))
loss=tf.reduce_sum(tf.square(output-out))/2+ploss
opti=tf.train.GradientDescentOptimizer(0.025).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tfg", sess.graph)
    for i in range(0,total_data):
        sm,p_loss=0,0
        ps=np.zeros(shape=(1,internal_dim))
        for j in range(0,data_len):
            sm+=data[i][j]
            inp_dict={inp:data[i][j].reshape(1,1),out:sm.reshape(1,1),ploss:p_loss,prev_state:ps}
            r=sess.run([opti,loss,state],feed_dict=inp_dict)
            p_loss+=r[1]
            ps=r[2]
        print(r[1])

