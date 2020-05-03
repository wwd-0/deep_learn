#单层（全连接层） 实现手写数字识别
import  tensorflow.compat.v1 as tfv1
import tensorflow.keras as kr
import  tensorflow as tf
def full_connected():
	# 1、建立数据的占位符 x [None,784] y_true[None,10]
	with tf.variable_scope("data"):
		x = tf.placeholder(tf.float32,[None,784])
		y_true = tf.placeholder(tf.int32,[None,10])

	# 2、建立一个全连接层的神经网络  w[784,10]  b[10]
	with tf.variable_scope("fc_model"):
		# 随机初始化权重和偏置
		weight = tf.Variable(tf.random_normal)

		bias = tf.Variable(tf.constant(0.0,shape=[10]))

		# 预测None个样本的输出结果 matrix[None,784] * [784,10] + [10] = [None,10]
	return None

if __name__ == "__main__":
	full_connected()