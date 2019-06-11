import functools
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as tf_input_data
import warnings
warnings.filterwarnings('ignore')

def doublewrap(function):
	@functools.wraps(function)
	def decorator(*args, **kwargs):
		if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
			return function(args[0])
		else:
			return lambda wrapee: function(wrapee, *args, **kwargs)
	return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
	attribute = '_cache_' + function.__name__#Get function name
	name = scope or function.__name__
	@property
	@functools.wraps(function)#Keep the original function
	def decorator(self):
		if not hasattr(self, attribute):#If the attribute not exist
			with tf.variable_scope(name, *args, **kwargs):#Add scope name
				setattr(self, attribute, function(self))
		return getattr(self, attribute)#otherwise return the attribute
	return decorator


class MNISTModel:
	def __init__(self, input_holder, target_holder, is_training, keep_prob):
		self.input_holder = input_holder
		self.target_holder = target_holder
		self.is_training = is_training
		self.keep_prob = keep_prob

		self.n_pixel = 784
		self.n_class = 10
		self.prediction
		self.optimize
		self.accuracy
		print('Model initialized!')

	@define_scope(initializer=tf.contrib.slim.xavier_initializer())
	def prediction(self):
		return None

	@define_scope
	def loss(self):
		return None

	@define_scope
	def optimize(self):
		return None

	@define_scope
	def accuracy(self):
		return None


class CNNModel(MNISTModel):
	@define_scope
	def prediction(self):
		logits = self.cnnLayer
		return tf.nn.softmax(logits)

	@define_scope
	def optimize(self):
		cur_step = tf.Variable(0, trainable=False)
		starter_lr = 0.0005
		lr = tf.train.exponential_decay(starter_lr, cur_step, 500, 0.96, staircase=True)
		optimizer = tf.train.AdamOptimizer(lr)
		cross_entropy = self.loss
		return optimizer.minimize(cross_entropy)

	@define_scope
	def loss(self):
		pred = self.prediction
		pred = tf.clip_by_value(pred, 1e-10, 1)
		cross_entropy = -tf.reduce_mean(self.target_holder * tf.log(pred))
		return cross_entropy

	@define_scope
	def accuracy(self):
		correct_prediction = tf.equal(tf.argmax(self.target_holder, 1), tf.argmax(self.prediction, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
		return accuracy

	@define_scope
	def correct_num(self):
		logits = self.cnnLayer
		labels = tf.argmax(self.target_holder, 1)
		eval_correct=tf.nn.in_top_k(logits, labels, 1)
		return eval_correct

	@define_scope
	def cnnLayer(self):
		x_image = tf.reshape(self.input_holder, [-1, 28, 28, 1])
		with tf.variable_scope('hidden1'):
			kernel_shape, bias_shape = [5, 5, 1, 32], [32] 
			h_pool1 = self.conv_relu_bn_pool(x_image, kernel_shape, bias_shape)
		with tf.variable_scope('hidden2'):
			kernel_shape, bias_shape = [5, 5, 32, 64], [64] 
			h_pool2 = self.conv_relu_bn_pool(h_pool1, kernel_shape, bias_shape)	
		with tf.variable_scope('fully_conn'):
			W_fc1 = self._get_variable(name='weights', shape=[7 * 7 * 64, 1024])
			b_fc1 = self._get_variable(name='bias', shape=[1024])
			h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
		with tf.variable_scope('softmax_layers'):
			W_fc2 = self._get_variable(name='weights', shape=[1024, 10])
			b_fc2 = self._get_variable(name='bias', shape=[10])
		logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2			
		return logits

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def _get_variable(self, name, shape, initializer=None, trainable=True):
		"A little wrapper around tf.get_variable"
		return tf.get_variable(name, shape=shape, initializer=initializer, trainable=trainable)

	def conv_relu_bn_pool(self, x, kernel_shape, bias_shape):
		initializer = tf.random_uniform_initializer(-0.01, 0.01)
		weights = self._get_variable(name='conv_weights', shape=kernel_shape, initializer=initializer)
		bias = self._get_variable(name='conv_bias', shape=bias_shape, initializer=initializer)
		conv = self.conv2d(x, weights)
		z = conv + bias
		bn = self.batch_normalization(z)
		relu = tf.nn.relu(bn)
		pool = self.max_pool_2x2(relu)
		return pool

	def batch_normalization(self, x):
		x_shape = x.get_shape()
		params_shape = x_shape[-1]
		axis = list(range(len(x_shape) - 1))
		beta = self._get_variable(name='beta', shape=params_shape, initializer=tf.zeros_initializer)
		gamma = self._get_variable(name='gamma', shape=params_shape, initializer=tf.ones_initializer)
		moving_mean = self._get_variable(name='moving_mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
		moving_variance = self._get_variable(name='moving_variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)
		mean, variance = tf.nn.moments(x, axis)
		decay = 0.95
		update_moving_mean = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
		update_moving_variance = tf.assign(moving_variance, moving_variance * decay + variance * (1- decay))
		if self.is_training:
			x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
		else:
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				x = tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, 0.001) 
		return x

def train_and_test():
	mnist = tf_input_data.read_data_sets('MNIST/', one_hot=True)
		
	n_train = mnist.train.images.shape[0]
	n_test = mnist.test.images.shape[0]
	n_pixel = 784
	n_class = 10

	batch_size = 64
	epochs = 10
	num_steps = int(n_train / batch_size)

	tf.reset_default_graph()
	graph = tf.Graph()

	with graph.as_default():
		initializer = tf.random_uniform_initializer(-0.01, 0.01)
		with tf.variable_scope('CNN', initializer=initializer) as scope:
			input_holder = tf.placeholder(tf.float32, [None, n_pixel])
			target_holder = tf.placeholder(tf.float32, [None, n_class])
			train_model = CNNModel(input_holder, target_holder, True, 0.5)
			tf.get_variable_scope().reuse_variables()
			test_model = CNNModel(input_holder, target_holder, False, 1)

	print('n_train =', n_train, 'num_steps =', num_steps)

	with tf.Session(graph=graph) as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		fout = open('log_with_bn.txt', 'w')
		fout.close()
		for epoch in range(epochs):
			loss, acc = 0, 0
			fout = open('log_with_bn.txt', 'a')
			for step in range(num_steps):
				batch_data, batch_labels = mnist.train.next_batch(batch_size)
				feed_dict = {input_holder:batch_data, target_holder:batch_labels}
				_, loss, acc = sess.run([train_model.optimize, train_model.loss, train_model.accuracy], feed_dict=feed_dict)

				if epoch == 0:
					fout.write(str(loss) + ' ' + str(acc) + '\n')
					print('step:', str(step), 'Loss:{:.5f}'.format(loss), 'Accuracy:{:.5f}'.format(acc))
			fout.close()
			print('epoch:', str(epoch), 'Loss:{:.5f}'.format(loss), 'Accuracy:{:.5f}'.format(acc))


		count = 0
		for _ in range(200):
			batch_data, batch_labels = mnist.test.next_batch(50)
			feed_dict = {input_holder:batch_data, target_holder:batch_labels}
			cp = sess.run(test_model.correct_num, feed_dict=feed_dict)
			count += np.sum(cp)
		print("Testing Accuracy：", count / n_ts)	






if __name__ == '__main__':
	train_and_test()












