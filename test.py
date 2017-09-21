import tensorflow as tf
import numpy as np
import random as rn
internal_dim=1
char=[]
chars=set()
with open("wordsEn.txt") as word_file:
    english_words = list(word.strip().lower() for word in word_file)
for w in range(0,len(english_words)):
    english_words[w]=english_words[w]+'.'
    chars.update(english_words[w])
char=list(chars)
num_char=len(char)
def chartovec(b):
    a=np.zeros((num_char,1))
    a[char.index(b)][0]=1
    return a
def vectochar(a):
    b=max(a)
    b=a.index(b)
    return char[b]
inp=tf.placeholder(name="inp",dtype=tf.float32,shape=(num_char,1))
out=tf.placeholder(name="out",dtype=tf.float32,shape=(1,num_char))
prev_state=tf.placeholder(name="prev_state",dtype=tf.float32,shape=(1,internal_dim))
b=tf.get_variable("b",initializer=tf.random_normal(shape=(1,internal_dim),mean=0,stddev=1))
prev_w=tf.get_variable("prev_w",initializer=tf.random_normal(shape=(internal_dim,internal_dim),mean=0,stddev=1))
in_w=tf.get_variable("in_w",initializer=tf.random_normal(shape=(num_char,internal_dim),mean=0,stddev=1))
out_w=tf.get_variable("out_w",initializer=tf.random_normal(shape=(internal_dim,num_char),mean=0,stddev=1))
state=tf.tanh(tf.matmul(tf.transpose(in_w),inp)+tf.matmul(prev_state,prev_w)+b)
output=tf.nn.softmax(tf.matmul(state,out_w))
loss=tf.reduce_mean(tf.square(output-out))
opti=tf.train.GradientDescentOptimizer(0.02).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tfg", sess.graph)
    for w in english_words:
        for i in range(0,3):
            s=np.zeros((1,internal_dim),dtype=np.float)
            for j in range(0,len(w)-1):
                inp_d={inp:chartovec(w[j]),out:np.transpose(chartovec(w[j+1])),prev_state:s}
                res=sess.run([opti,loss,out,state],feed_dict=inp_d)
                s=res[3]
        print(res[1])
