import tensorflow.compat.v1 as tf
import tensorflow as tf2
from tensorflow.keras import datasets  # 导入经典数据集加载模块
tf.disable_eager_execution()

#定义一个初始化权重函数
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
    return w

#定义一个初始化偏置函数
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0,shape=shape))
    return b

def model():
    """
    自定义的卷积模型
    :return: None
    """
    #1、 准备数据占位符 x [None,784] 样本数量后续填充 784个特征 y_true[None,10] 十个真实值
    with tf.Variable_scope("data"):
        x = tf.placeholder(tf.float32,[None,784])
        y_true = tf.placeholder(tf.int32,[None,10])

    # 2、一卷积层
    with tf.variable_scope("conv1"):
        #初始化权重 1 个通道 32 个 黑白图片
        w_con1 = weight_variables([5,5,1,32])  # 权重 即 卷积窗口大小
        #初始化偏置 32个
        b_con1 = bias_variables([32])

        # 对x进行形状的改变 [None,784] [None,28,28,1]
        # reshape 第一个参数不能为None 用-1占位
        x_reshape = tf.reshape(x,[-1,28,28,1])

        # [None,28,28,1]-------->> [None,28,28,32] 激活函数 采用SAME 卷积后输出特征矩阵大小与原图大小相同
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape,w_con1,strides=[1,1,1,1],padding="SAME") + b_con1)

        # 池化层 2*2 ，stride [None,28,28,32]-------->> [None,14,14,32] # 这里的same不是指输入和输出大小相等 只是指定了图像边缘的取样规则
        x_pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],stride=[1,2,2,1],padding="SAME")

    # 3、二卷积层
    with tf.variable_scope("conv2"):
        # 随机初始化权重 和偏置 64 个权重 和偏置
        w_con2 = weight_variables([5,5,32,64])
        #初始化偏置 32个
        b_con2 = bias_variables([64])

        # [None,14,14,32]-------->> [None,14,14,64] 激活函数 采用SAME 卷积后输出特征矩阵大小与原图大小相同
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_con2, strides=[1, 1, 1, 1], padding="SAME") + b_con2)

        # 池化层 2*2 ，stride [None,14,14,64]-------->> [None,7,7,64] # 这里的same不是指输入和输出大小相等 只是指定了图像边缘的取样规则
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding="SAME")

    # 4、全连接层 [None,7,7,64]--->[None,7*7*64]*[None,7*7*64,10]+[10] =[None,10]
    with tf.variable_scope("conv3"):
        # 随机初始化权重 和偏置 10 个权重 和偏置
        w_fc = weight_variables([7*7*64, 10])
        # 初始化偏置 10个
        b_fc = bias_variables([10])

        #修改形状
        x_fc_reshape = tf.reshape(x_pool2,[-1,7*7*64])

        #进行矩阵运算得出每个样本的10个结果
        y_predict = tf.matmul(x_fc_reshape,w_fc) + b_fc

    return x,y_true,y_predict

def conv_fc():
    # 获取真实 的数据
    (x, y), (x_test, y_test) = datasets.mnist.load_data()  # 返回数组的形状
   # ds_train = tfds.load(name="../ANN/minst/", split="train",one_hot=True)
    # 定义模型 进行输出
    x,y_true,y_pridect = model()

    #交叉熵损失计算
    with tf.variable_scope("soft_cross"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pridect))
    # 梯度下降求损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0,1).minimize(loss)
    # 计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true,1),tf.argmax(y_pridect,1))

        #equal_list None 个样本 [1,0,1,0,1,1,.....]
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))
    #定义一个初始化变量op
    init_op = tf.global_variables_initializer()

    #开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(1000):
            mnist_x,mnist_y = mnist.train.next_batch(50)
            # 运行train_op训练
            sess.run(train_op,feed_dict={x:mnist_x,y_true:mnist_y})

    return None

if __name__ == "__main__":
    conv_fc()
    print("=============end")