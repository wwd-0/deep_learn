import  tensorflow as tf
import  tensorflow.compat.v1 as tfv1
# tensorflow 读取数据学习
tfv1.disable_eager_execution()

#tensorflow 队列操作过程
def dataadd():
	# 模拟 一下同步先处理数据 然后才能取数据 训练
	# tensorflow 运行操作有依赖性 本列中data依赖en_q
	# 1、首先定义 队列
	# 2、定义一些读取数据的过程

	# 1、定义一个队列
	Q = tfv1.FIFOQueue(3, tf.float32)

	# 放如一些 数据
	enq_many = Q.enqueue_many([[0.1, 0.2, 0.3],])

	# 2、定义一些处理数据的 取数据的过程  取数据 +1 入队列
	out_q = Q.dequeue()

	data = out_q + 1

	en_q = Q.enqueue(data)

	with tfv1.Session() as sess:
		# 初始化队列
		sess.run(enq_many)

		# 处理数据
		for i in range(100):
			sess.run(en_q)

		# 训练数据
		for i in range(Q.size().eval()):
			print(sess.run(out_q))

# tensorflow多线程操作
def tenthread():
	# 1、定义一个队列
	Q = tfv1.FIFOQueue(1000, tf.float32)

	# 2、定义要做的事情 循环+1 ，放入队列中
	var = tf.Variable(0.0)

	#实现自增 tf.assign_add
	data = tfv1.assign_add(var,tf.constant(1.0))

	en_q = Q.enqueue(data)

	# 3、定义队列管理器op,指定多少个子线程，子线程该干什么事情
	qr = tfv1.train.QueueRunner(Q,enqueue_ops=[en_q]*4)

	#3、初始化变量op
	init_op = tfv1.global_variables_initializer()

	with tfv1.Session() as sess:
		# 初始化队列
		sess.run(init_op)

		# 开启线程管理器
		coord = tf.train.Coordinator()

		# 开启线程
		threads = qr.create_threads(sess,coord=coord,start=True)

		# 主线程 不断读取数据训练
		for i in range(500):
			print(sess.run(Q.dequeue()))

		# 回收子线程
		coord.request_stop()

		coord.join(threads)

if __name__ == "__main__":
	tenthread()