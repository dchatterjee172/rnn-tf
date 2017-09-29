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
prev_w=tf.get_variable("prev_w",initializer=tf.random_normal(shape=(internal_dim,internal_dim),mean=.5,stddev=1))
in_w=tf.get_variable("in_w",initializer=tf.random_normal(shape=(1,internal_dim),mean=.5,stddev=1))
out_w=tf.get_variable("out_w",initializer=tf.random_normal(shape=(internal_dim,1),mean=.5,stddev=1))
state=tf.nn.elu(tf.matmul(inp,in_w)+tf.matmul(prev_state,prev_w))
output=tf.matmul(state,out_w)
loss=tf.reduce_sum(tf.square(out-output)/2)+ploss
pre=tf.placeholder(name="dstatepre",dtype=tf.float32,shape=(1,internal_dim))
dout_w=tf.gradients(loss,out_w)
dstate=tf.gradients(loss,state)+pre
dprestate=tf.gradients(state,prev_state,grad_ys=dstate[0]) 
dprev_w=tf.gradients(state,prev_w,grad_ys=dstate[0])
din_w=tf.gradients(state,in_w,grad_ys=dstate[0])
tdo=tf.placeholder(name="gradientdout_w",dtype=tf.float32,shape=(internal_dim,1))
tdh=tf.placeholder(name="gradientdh_w",dtype=tf.float32,shape=(internal_dim,internal_dim))
tdi=tf.placeholder(name="gradientdin_w",dtype=tf.float32,shape=(1,internal_dim))
lr=tf.placeholder(name="lrate",dtype=tf.float32,shape=())
change_o=tf.assign(out_w,out_w-tdo*lr)
change_h=tf.assign(prev_w,prev_w-tdh*lr)
change_i=tf.assign(in_w,in_w-tdi*lr)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tfg", sess.graph)
    lrate=1
    for i in range(0,total_data):
        sm=np.zeros(shape=(data_len,))
        closs=np.zeros(shape=(data_len,))
        states=np.zeros(shape=(data_len,1,internal_dim))
        tdout_w,tdprev_w,tdin_w=np.zeros(shape=(internal_dim,1)),np.zeros(shape=(internal_dim,internal_dim)),np.zeros(shape=(1,internal_dim))
        dprestates=np.zeros(shape=(data_len,1,internal_dim))
        for j in range(0,data_len):
            if j>0:
                sm[j]=data[i][j]+sm[j-1]
                inp_dict={inp:data[i][j].reshape(1,1),out:sm[j].reshape(1,1),ploss:closs[j-1],prev_state:states[j-1]}
            else:
                sm[j]=data[i][j]
                inp_dict={inp:data[i][j].reshape(1,1),out:sm[j].reshape(1,1),ploss:0,prev_state:np.zeros(shape=(1,internal_dim))}
            r=sess.run([loss,state,output],feed_dict=inp_dict)
            closs[j]=r[0]
            states[j]=r[1]
        for j in reversed(range(0,data_len)):
            if j==0:
                inp_dict={inp:data[i][j].reshape(1,1),out:sm[j].reshape(1,1),ploss:0,prev_state:np.zeros(shape=(1,internal_dim)),pre:dprestates[j+1]}
            elif j<data_len-1:
                inp_dict={inp:data[i][j].reshape(1,1),out:sm[j].reshape(1,1),ploss:closs[j-1],prev_state:states[j-1],pre:dprestates[j+1]}
            else:
                inp_dict={inp:data[i][j].reshape(1,1),out:sm[j].reshape(1,1),ploss:closs[j-1],prev_state:states[j-1],pre:np.zeros(shape=(1,internal_dim))}
            m=sess.run([din_w,dout_w,dprev_w,dprestate],feed_dict=inp_dict)
            tdout_w+=m[1][0]
            tdin_w+=m[0][0]
            tdprev_w+=m[2][0]
            dprestates[j]=m[3][0]
            tdprev_w=np.clip(tdprev_w,-.5,.5)
            tdin_w=np.clip(tdin_w,-.5,.5) 
            tdout_w=np.clip(tdout_w,-.5,.5) 
        inp_dict={lr:lrate,tdo:tdout_w,tdh:tdprev_w,tdi:tdin_w}
        x=sess.run([change_o,change_i,change_h],feed_dict=inp_dict)
        print(r[0],sm[data_len-1],r[2][0])
        lrate*=.99
