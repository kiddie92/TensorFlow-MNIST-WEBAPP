import numpy as np
import tensorflow as tf
from flask import Flask,jsonify,render_template,request
from mnist import model

x=tf.placeholder("float",[None,784])
# conan_zzh
config = tf.ConfigProto(allow_soft_placement=True)
# 最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# gpu资源按需增加
config.gpu_options.allow_growth = True
###

sess=tf.Session(config=config)

with tf.variable_scope("regression"):
    y1,variables=model.regression(x)

saver=tf.train.Saver(variables)
saver.restore(sess,"mnist/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob=tf.placeholder(tf.float32)
    y2,variables=model.convolutional(x,keep_prob)
saver=tf.train.Saver(variables)
saver.restore(sess,'mnist/data/convolutional.ckpt')

def regression(input):
    return sess.run(y1,feed_dict={x:input}).flatten().tolist()

def convolutional(input):
    return sess.run(y2,feed_dict={x:input,keep_prob:1.0}).flatten().tolist()

app=Flask(__name__)
@app.route('/api/mnist',methods=['post'])
def mnist():
    input=((255-np.array(request.json,dtype=np.uint8))/255.0).reshape(1,784)
    output1=regression(input)
    output2=convolutional(input)
    return jsonify(results=[output1,output2])

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.debug=True
    app.run()
