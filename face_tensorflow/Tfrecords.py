import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import tensorflow.compat.v1.app as tfapp

# 定义cifar的数据等命令行参数
FLAGS = tfapp.flags.FLAGS
tfapp.flags.DEFINE_string("tfpath","./face.tfrecords","文件的目录")

class TFRecords:
    def ___init__(self):
        pass
    def wirte_tfrecords(sefl,image_batch,label_batch):
        """
        将图片特征值和目标值存入tfrecords

        """
        #1 建立tfRecord存储器
        write = tfv1.python_io.TFRecordWriter(FLAGS.tfpath)
        #2 遍历所有样本
        

        return None
if __name__ =="__main__":
    tfrecord = TFRecords()
    tfrecord.wirte_tfrecords(None,None)