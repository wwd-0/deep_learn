import tensorflow.compat.v1 as tfv1
import tensorflow as tf
tfv1.disable_eager_execution()
FLAGS = tfv1.app.flags.FLAGS

tfv1.app.flags.DEFINE_string("job_name","worker","启动服务的类型 ps or worker")
tfv1.app.flags.DEFINE_integer("task_index",0 ,"指定ps或者worker中的哪一台服务器以task：0")

def main(argv):
	#定义全局计数的op，给钩子列表当中的训练步数使用
	global_step = tfv1.train.get_or_create_global_step()

	# 指定集群描述对象，ps or work
	cluster = tfv1.train .ClusterSpec({"ps":["192.168.1.103:2223"],"worker":["192.168.1.104:2222"]})

	# 创建不同的服务，ps worker
	server = tfv1.train.Server(cluster,job_name = FLAGS.job_name,task_index = FLAGS.task_index)

	# 指定不同服务做不同的事情， ps：去更新保存参数 worker：指定设备去运行模型计算
	if FLAGS.job_name == "ps":
		# 参数服务器什么都不用干，是需要等待woker传递参数
		server.join()
	else:
		woker_device = "/job:worker/task:0/cpu:0/"

		# 可以指定设备运行
		with tfv1.device(tfv1.train.replica_device_setter(
			worker_device = woker_device,
			cluster = cluster
		)):
			#做一个简单的加法运算
			x = tfv1.Variable([[1, 2, 3, 4]])
			w = tfv1.Variable([[1], [2], [3], [4]])

			mat = tfv1.matmul(x,w)

		# 创建分布式会话
		with tfv1.train.MonitoredTrainingSession(
			master = "grpc://192.168.1.104:2222",#指定主worker
			is_chief = (FLAGS.task_index == 0), # 判断是否是主worker
			config = tfv1.ConfigProto(log_device_placement=True,device_filters=["/job:ps","/job:worker/task:0"] ), #打印设备信息
			hooks = [tfv1.train.StopAtStepHook(last_step = 200)]
		) as mon_sess :
			while not mon_sess.should_stop():
				print(mon_sess.run(mat))

if __name__ =="__main__":
	tfv1.app.run()
