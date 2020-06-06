import numpy as np
import  tensorflow as tf
import  tensorflow.compat.v1 as tfv1
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tensorflow 读取数据学习  -- 读取csv文件
tfv1.disable_eager_execution()

# 定义cifar的数据等命令行参数
FLAGS = tfv1.app.flags.FLAGS
tfv1.app.flags.DEFINE_string("cifar_dir","./","文件的目录")
tfv1.app.flags.DEFINE_string("train_dir","./train/","文件的目录")
sess = tfv1.Session()
class LoadData:
    def __init__(self):
        pass

    def LoadTrainPathAndLabel(self,filelist):
        """
        读取txt文件
        :param filelist: 文件列表 + 名字的列表
        :return: 读取的内容
        """
        filelt = np.loadtxt(filelist,dtype=str,delimiter=' ')  # 最普通的loadtxt
        image, label = np.char.add([FLAGS.train_dir], filelt[:, 0]), filelt[:, 1]

        deleteList = []
        for j in range(len(image)):
            my_file = Path(image[j])
            if not my_file.is_file():
                deleteList.append(j)
                print(image[j])

        print(image.size)
        image = np.delete(image,deleteList, axis=0)  # 删除
        label = np.delete(label,deleteList, axis=0)  # 删除
        print(image.size)
        return  image,label

    def imageread(self,image,label):
        """
        读取图片文件 并转换成张量
        :param filelist: 文件列表 + 名字的列表
        :return: 读取的内容
        """
        image = tfv1.cast(image, tf.string)
        label = tfv1.cast(label, tf.string)
        # 1、构造文件队列
        file_queue = tfv1.train.slice_input_producer([image,label])

        # 2、构造csv阅读器读取队列数
        value = tfv1.read_file(file_queue[0])

        label = file_queue[1]

        # 3、对读取的文件进行解码
        image = tfv1.image.decode_jpeg(value)

        # 4、处理图片统一大小
        image_resize = tfv1.image.resize_images(image, [227, 227])

        #  一定要把样本的形状固定 [200,200,3] 在批处理的 时候要求所有数据必须定义
        image_resize.set_shape([227, 227, 3])

        # 5、进行批处理
        image_batch,label_batch = tfv1.train.batch([image_resize,label], batch_size=100, num_threads=1, capacity=100)

        return image_batch,label_batch


if __name__ == "__main__":
    imageload = LoadData()

    image,label =  imageload.LoadTrainPathAndLabel("train.txt")

    if(image.size>0):

        image_batch, lable_batch = imageload.imageread(image,label)
        # 开启会话运行

        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读取文件的线程
        threads = tfv1.train.start_queue_runners(sess, coord=coord)

        try:
            while not coord.should_stop():
                print(sess.run([image_batch, lable_batch]))
        except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
            print("done! now lets kill all the threads……")
        finally:
            # 协调器coord发出所有线程终止信号
            coord.request_stop()

        coord.join(threads)

