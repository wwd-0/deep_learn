import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tfv1.disable_eager_execution()

def myregression():
	"""自实现一个线程回归
	:return None
	"""
	#1、准备数据 x 特征 [100,1] y 目标值[100]
	x = tfv1.random_normal([100,1],mean=1.75,stddev=0.5 ,name='x_data')

	y_true = tf.matmul(x , [[0.7]]) + 0.8

	# 2、建立线性回归模型 1 个特征 1 个权重 一个偏置 y = xw + b
	#随机给一个权重和偏置，让他去计算损失，然后在当前状态下优化
	# 用变量定义才能优化
	weight = tf.Variable(tfv1.random_normal([1,1],mean=0.0,stddev=1.0),name="w")
	bias =  tf.Variable(0.0,name="b")
	y_predict = tf.matmul(x,weight) + bias

	#3、建立损失函数，均方误差
	loss = tf.reduce_mean(tf.square(y_true - y_predict))
	print(loss)

	#3、梯度下降优化损失 leaning_rate :0c ~ 1 2 3 5 7 10
	train_op = tfv1.train.GradientDescentOptimizer(0.1).minimize(loss)

	#定义一个初始化变量的op
	init_op = tfv1.global_variables_initializer()

	# 通过会话运行程序
	with tfv1.Session() as sess:
		#初始化变量
		sess.run(init_op)

		#打印随机最先初始化的权重和偏置
		print("随机初始化的参数权重为：%f,偏置为：%f"  % (weight.eval(),bias.eval()))
		sess.run(train_op)

		#循环训练 运行优化
		#for i in range(1000):
		sess.run(train_op)
#		print("第%d次优化的参数权重为%f,偏置为:%f" % (i,weight.eval(),bias.eval()))
if __name__ == "__main__":
	myregression()
