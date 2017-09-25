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
prev_w=tf.get_variable("prev_w",initializer=tf.random_normal(shape=(internal_dim,internal_dim),mean=1,stddev=1))
in_w=tf.get_variable("in_w",initializer=tf.random_normal(shape=(1,internal_dim),mean=1,stddev=1))
out_w=tf.get_variable("out_w",initializer=tf.random_normal(shape=(internal_dim,1),mean=1,stddev=1))
state=tf.nn.elu(tf.matmul(inp,in_w)+tf.matmul(prev_state,prev_w))
output=tf.nn.elu(tf.matmul(state,out_w))
loss=tf.reduce_sum(tf.square(output-out))+ploss
dout_w=tf.gradients(loss,out_w)
dprev_w=tf.gradients(loss,prev_w)
din_w=tf.gradients(loss,in_w)
tdo=tf.placeholder(name="gradientdout_w",dtype=tf.float32,shape=(internal_dim,1))
tdh=tf.placeholder(name="gradientdh_w",dtype=tf.float32,shape=(internal_dim,internal_dim))
tdi=tf.placeholder(name="gradientdin_w",dtype=tf.float32,shape=(1,internal_dim))
lr=tf.placeholder(name="lrate",dtype=tf.float32,shape=())
out_w=out_w-tdo*lr
prev_w=prev_w-tdh*lr
in_w=in_w-tdi*lr
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tfg", sess.graph)
    lrate=.03
    for i in range(0,total_data):
        sm=0
        closs=np.zeros(shape=(data_len,))
        states=np.zeros(shape=(data_len,1,internal_dim))
        tdout_w,tdprev_w,tdin_w=np.zeros(shape=(internal_dim,1)),np.zeros(shape=(internal_dim,internal_dim)),np.zeros(shape=(1,internal_dim))
        for j in range(0,data_len):
            sm+=data[i][j]
            if j>0:
                inp_dict={inp:data[i][j].reshape(1,1),out:sm.reshape(1,1),ploss:closs[j-1],prev_state:states[j-1]}
            else:
                inp_dict={inp:data[i][j].reshape(1,1),out:sm.reshape(1,1),ploss:0,prev_state:np.zeros(shape=(1,internal_dim))}
            r=sess.run([loss,state,output,dout_w,dprev_w,din_w],feed_dict=inp_dict)
            closs[j]=r[0]
            states[j]=r[1]
            tdout_w+=r[3][0]
            tdprev_w+=r[4][0]
            tdin_w+=r[5][0]
        inp_dict={lr:lrate,tdo:tdout_w,tdh:tdprev_w,tdi:tdin_w}
        sess.run([in_w,out_w,prev_w],feed_dict=inp_dict)
        del closs,states,tdout_w,tdprev_w,tdin_w
        print(r[0])
