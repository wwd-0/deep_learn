import tensorflow.compat.v1 as tf
import  tensorflow as tf2
import numpy as np
class CNN:
    def __init__(self):
        pass

    # 定义一个初始化偏置函数
    def bias_variables(self,shape):
        return  tf.Variable(tf.constant(0.0, shape=shape))

    # 定义一个初始化权重函数
    def weight_variables(self,shape):
        return tf.Variable(tf.random_normal(shape=shape))

    def conv(self,shape):
        weight  = self.weight_variables(shape)
        bais = self.bias_variables(shape)
        print(weight)
        return weight
