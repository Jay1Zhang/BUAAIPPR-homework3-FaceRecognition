import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

from dataset import Dataset

# 生成数据集
dataset = Dataset('Pose05')
dataset.load()
trainset, testset = dataset.gen_cnn_dataset()

# 定义并训练CNN模型
num_people = 68

data_input = tf.placeholder(tf.float32,[None, 64, 64, 1])
label_input = tf.placeholder(tf.float32,[None, num_people])

#实现CNN卷积神经网络，并测试最终训练样本实现的检测概率
#tf.layer方法可以直接实现一个卷积神经网络的搭建
#通过卷积方法实现
layer1 = tf.layers.conv2d(inputs=data_input, filters = 32,kernel_size=2,
                          strides=1,padding='SAME',activation=tf.nn.relu)
#实现池化层，减少数据量，pool_size=2表示数据量减少一半
layer1_pool = tf.layers.max_pooling2d(layer1,pool_size=2,strides=2)
#第二层设置输出，完成维度的转换，以第一次输出作为输入，建立n行的32*32*32输出
layer2 = tf.reshape(layer1_pool,[-1,32*32*32])
#设置输出激励函数
layer2_relu = tf.layers.dense(layer2, 1024, tf.nn.relu)
#完成输出，设置输入数据和输出维度
output = tf.layers.dense(layer2_relu, num_people)

#建立损失函数
loss = tf.losses.softmax_cross_entropy(onehot_labels=label_input,logits=output)
#使用梯度下降法进行训练
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#定义检测概率
accuracy = tf.metrics.accuracy(
    labels=tf.arg_max(label_input, 1), predictions=tf.arg_max(output, 1))[1]

#对所有变量进行初始化
init = tf.group(
    tf.global_variables_initializer(),tf.local_variables_initializer(),tf.local_variables_initializer())
#定义for循环，完成样本的加载和数据训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(0, 2):
        print('e: ' + str(i))
        #完成数据加载并计算损失函数和训练值
        sess.run([train,loss],feed_dict={data_input: trainset['data'],
                                         label_input: trainset['label']})
        acc = sess.run(accuracy,feed_dict={data_input: testset['data'],
                                           label_input: testset['label']})
#打印当前概率精度
    print('acc:%.2f',acc)
