import  tensorflow as tf
import  tensorflow.compat.v1 as tfv1
import  numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tensorflow 读取数据学习  -- 读取csv文件

# 定义cifar的数据等命令行参数
FLAGS = tfv1.app.flags.FLAGS
# 正样本路径
tfv1.app.flags.DEFINE_string("pimage","../../face_tensorflow/train/00/","文件的目录")

#负样本路径
FLAGS = tfv1.app.flags.FLAGS
tfv1.app.flags.DEFINE_string("nimage","../../face_tensorflow/train/11/","文件的目录")

imagewidth = 227
imageheight = 227

#读取文件
def readImage(fileName,label):
	"""
	:param fileName:
	:param label:
	:return:
	"""
	#读取图片文件
	value = tfv1.read_file(fileName)
	#图片数据解码
	imagedata = tf.image.decode_jpeg(value, channels=3)
	#处理图片统一大小
	imagedata = tfv1.image.resize_images(imagedata, [imagewidth, imageheight])
	#  一定要把样本的形状固定 [227,227,3] 在批处理的 时候要求所有数据必须定义
	imagedata.set_shape([227, 227, 3])

	image = tf.cast(imagedata, tf.float32)
	return image, label

#tf2.* 版本数据批处理数据
def imageread_2():
	"""
	:param filelist: 图片路径列表
	:param labellist: 对应标签列表
	:return:
	"""
	# 1、找到文件，放入列表 路径+名字 -》列表
	pfilelist = os.listdir(FLAGS.pimage)
	nfilelist = os.listdir(FLAGS.nimage)

	filelist = [os.path.join(FLAGS.pimage, file) for file in pfilelist]
	plen = len(filelist)
	labellist = [0] * plen

	filelist.extend([os.path.join(FLAGS.nimage, file) for file in nfilelist])
	labellist.extend([1] * (len(filelist) - plen))

	#切片:q
	dataset = tf.data.Dataset.from_tensor_slices((filelist,labellist))

	#打乱顺序
	dataset = dataset.shuffle(100)
	# 循环1次
	dataset = dataset.repeat(1)
	# num_parallel_calls一般设置为cpu内核数量
	dataset = dataset.map(readImage, num_parallel_calls=2)
	# 指定batch_size 大小
	dataset = dataset.batch(50)
	#dataset = dataset.prefetch(2)  # software pipelining 机制
	return dataset

"""
#tf1.* 版本数据批处理数据
def imageread(filelist):
	
	#读取图片文件 并转换成张量
	#:param filelist: 文件列表 + 名字的列表
	#:return: 读取的内容
	
	# 1、构造文件队列
	file_queue = tf.data.Dataset.from_tensor_slices(filelist)

	# 2、构造csv阅读器读取队列数据
	reader = tfv1.WholeFileReader()
	key, value = reader.read(file_queue)

	# 3、对读取的文件进行解码
	image = tfv1.image.decode_jpeg(value)

	# 4、处理图片统一大小
	image_resize = tfv1.image.resize_images(image,[227,227])

	#  一定要把样本的形状固定 [200,200,3] 在批处理的 时候要求所有数据必须定义
	image_resize.set_shape([227,227,3])

	# 5、进行批处理
	imgae_batch = tfv1.train.batch([image_resize], batch_size=10, num_threads=1, capacity=1160)

	return imgae_batch
"""
if __name__ =="__main__":

	dataset = imageread_2()

	i=0
	for v in dataset:
		i += 1
		print(v)
	print(i)



"""
	# 开启会话运行
	with   tfv1.Session() as sess:
		#init_op = tfv1.group(tfv1.global_variables_initializer(), tfv1.local_variables_initializer())
		#sess.run(init_op)

		# 从 dataset 中实例化一个iterator，该 iterator 具有 one shot iterator 特性，
		# 即只能从头到尾读取一次

		# 定义一个线程协调器
		coord = tf.train.Coordinator()

		# 开启读取文件的线程
		threads = tfv1.train.start_queue_runners(sess,coord=coord)

		i = 0
		while not coord.should_stop():
			print(sess.run([dataset]))
			if i >3:
				coord.request_stop()

		coord.join(threads)
"""
