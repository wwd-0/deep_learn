import  tensorflow as tf
import  tensorflow.compat.v1 as tfv1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tensorflow 读取数据学习  -- 读取csv文件
tfv1.disable_eager_execution()

def csvread(filelist):
	"""
	读取csv文件
	:param filelist: 文件列表 + 名字的列表
	:return: 读取的内容
	"""

	# 1、构造文件队列
	file_queue = tfv1.train.string_input_producer(filelist)

	# 2、构造csv阅读器读取队列数据
	reader = tfv1.TextLineReader()
	key ,value = reader.read(file_queue)

	# 3、对每行数据进行解码
	# record_defaults:指定每一个样本的每一列的类型，指定默认值
	records = [["None"],["None"]]

	example,lable = tfv1.decode_csv(value,record_defaults=records )

	# 4、想要读多个数据，就需要批处理
	example_batch,lable_batch = tfv1.train.batch([example,lable],batch_size=4,num_threads=1,capacity=5)

	return example_batch,lable_batch

if __name__ =="__main__":
	# 1、找到文件，放入列表 路径+名字 -》列表
	file_name = os.listdir("./csvdata/")

	filelist = [os.path.join("./csvdata/",file) for file in file_name]

	example,lable = csvread(filelist)

	# 开启会话运行
	with   tfv1.Session() as sess:
		# 定义一个线程协调器
		coord = tf.train.Coordinator()

		# 开启读取文件的线程
		threads = tfv1.train.start_queue_runners(sess,coord=coord)

		for i in range(3):
			print(sess.run([example,lable]))
			print("===================")

		coord.request_stop()

		coord.join(threads)
