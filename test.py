import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 实现一个加法运算
# tfv1.disable_eager_execution()
g = tf.Graph()
print(g)

with g.as_default():
	c = tf.constant(5.0)
	print(c.graph)

a = tf.constant(5.0)
b = tf.constant(5.0)
print(a,b)

sum1 = tf.add(a,b)
print(sum1)

graph = tfv1.get_default_graph()
print(graph)

# 只能运行一个图 可以在会话中指定图去运行
# 只要有会话的上下文环境 就可以使用方便eval()
with tfv1.Session(graph=g,config=tfv1.ConfigProto(log_device_placement=True)) as sess:
	print(sess.run(c))
	print(c.eval())
