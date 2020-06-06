import  tensorflow as tf
import  tensorflow.compat.v1 as tfv1
import  numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tensorflow 读取数据学习  -- 读取csv文件

# 定义cifar的数据等命令行参数
FLAGS = tfv1.app.flags.FLAGS
# 正样本路径
tfv1.app.flags.DEFINE_string("pimage","../train/00/","文件的目录")

#负样本路径
FLAGS = tfv1.app.flags.FLAGS
tfv1.app.flags.DEFINE_string("nimage","../train/11/","文件的目录")

pfilelist = os.listdir(FLAGS.pimage)
nfilelist = os.listdir(FLAGS.nimage)

labellist = []
filelist= [os.path.join(FLAGS.pimage,file) for file in pfilelist]

labellist = [0]*len(filelist)
filelist__= [os.path.join(FLAGS.nimage, file) for file in nfilelist]

labellist.extend([1]*len(filelist__))

dataset = tf.data.Dataset.range(100)

for v in dataset:
	print(v)
#data = tf.data.Dataset.from_tensor_slices((filelist,labellist))
#print(data)  # 输出张量的信
