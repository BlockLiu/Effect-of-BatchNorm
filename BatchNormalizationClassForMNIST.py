#Below wrapper code was referred from https://github.com/richardsun-voyager/DeepLearningModelsforMNIST#Import required packages
import warnings
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline
#Load MNIST dataset
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
warnings.filterwarnings("ignore")

plt.imshow(mnist.train.images[5].reshape([28, 28]))
print(mnist.train.labels[5])

n_tr = mnist.train.images.shape[0]# number of training samples
n_ts = mnist.test.images.shape[0]#number of testing samples
n_pixel = mnist.train.images.shape[1]


#Below wrapper code was referred to http://danijar.com/structuring-your-tensorflow-models/
import functools
def doublewrap(function):
	"""
	A decorator decorator, allowing to use the decorator to be used without
	parentheses if not arguments are provided. All arguments must be optional.
	"""
	@functools.wraps(function)
	def decorator(*args, **kwargs):
		if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
			return function(args[0])
		else:
			return lambda wrapee: function(wrapee, *args, **kwargs)
	return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
	"""
	A decorator for functions that define TensorFlow operations. The wrapped
	function will only be executed once. Subsequent calls to it will directly
	return the result so that operations are added to the graph only once.

	The operations added by the function live within a tf.variable_scope(). If
	this decorator is used with arguments, they will be forwarded to the
	variable scope. The scope name defaults to the name of the wrapped
	function.
	"""
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


#Define a base class in order to be inherited
class MnistModel:
	'''Define a basic model for MNIST image classification, the model
	Provides graph structure of tensorflow'''
	
	def __init__(self, input_holder, target_holder, is_training, keep_prob):
		self.input_holder = input_holder
		self.target_holder = target_holder
		self.is_training = is_training
		self.num_pixel = 784
		self.num_class = 10
		self.keep_prob = keep_prob
		self.prediction
		self.optimize
		self.accuracy
		print('Model Initialized!') 
	
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


class CnnMnistModel(MnistModel):
	#重构某些函数
	
	@define_scope
	def prediction(self):
		#定义权重和偏置参数变量
		logits = self.cnnLayer
		return tf.nn.softmax(logits)

	@define_scope
	def optimize(self):
		 #Gradient Descending Optimizer
		#cur_step = tf.get_variable(name='step', initializer=0, dtype=tf.int32, trainable=False)
		cur_step = tf.Variable(0, trainable=False)
		# count the number of steps taken
		starter_learning_rate = 0.0005
		learning_rate = tf.train.exponential_decay(starter_learning_rate, cur_step, 500, 0.96, staircase=True)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		cross_entropy = self.loss
		#optimizer = tf.train.MomentumOptimizer(1e-4, 0.9)
		return optimizer.minimize(cross_entropy)
	
	@define_scope
	def loss(self):
		pred = self.prediction
		pred = tf.clip_by_value(pred, 1e-10, 1)
		cross_entropy = -tf.reduce_mean(self.target_holder*
									   tf.log(pred))
		return cross_entropy

	@define_scope
	def accuracy(self):
		correct_prediction = tf.equal(tf.argmax(self.target_holder,1), 
									  tf.argmax(self.prediction,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		return accuracy
	
	@define_scope
	def correct_num(self):
		'''Count correct predictions for testing part'''
		#labels = tf.one_hot(self.label_holder, self.num_class, 1, 0)
		#correct_prediction = tf.equal(tf.argmax(self.target_holder,1), 
									#tf.argmax(self.prediction,1))
		#eval_correct = tf.reduce_sum(tf.cast(correct_prediction, "float"))
		logits = self.cnnLayer
		labels = tf.argmax(self.target_holder, 1)
		eval_correct=tf.nn.in_top_k(logits,labels,1)
		return eval_correct
	
	@define_scope
	def cnnLayer(self):
		#Note, we need to share weights here, so variable_scope 
		#should be specified
		x_image = tf.reshape(self.input_holder, [-1,28,28,1])
		#First Conv
		with tf.variable_scope('hidden1'):
			kernel_shape, bias_shape = [5, 5, 1, 32], [32] 
			h_pool1 = self.conv_relu_bn_pool(x_image, kernel_shape, bias_shape)
			#Second variable_scope
		with tf.variable_scope('hidden2'):
			kernel_shape, bias_shape = [5, 5, 32, 64], [64] 
			h_pool2 = self.conv_relu_bn_pool(h_pool1, kernel_shape, bias_shape)
	
		#Fully Connected Layer
		with tf.variable_scope('fully_conn'):
			W_fc1 = self._get_variable(name='weights', shape=[7 * 7 * 64, 1024])
			b_fc1 = self._get_variable(name='bias', shape=[1024])
			h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
			
		#Dropout, to prevent against overfitting	  
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
		#Softmax Layer
		with tf.variable_scope('softmax_layers'):
			W_fc2 = self._get_variable(name='weights', shape=[1024, 10])
			b_fc2 = self._get_variable(name='bias', shape=[10])
		logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
			
		return logits
	
	#卷积函数
	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	
	#池化函数
	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							  strides=[1, 2, 2, 1], padding='SAME')

	

	def _get_variable(self, name,
				  shape,
				  initializer=None,
				  trainable=True):
		"A little wrapper around tf.get_variable"
		return tf.get_variable(name,
						   shape=shape,
						   initializer=initializer,
						   trainable=trainable)

	def conv_relu_bn_pool(self, x, kernel_shape, bias_shape):
		'''
		Create variables
		Run convolutions and batch normalization
		Run relu and maxpool
		'''
		# Create variable named "weights".
		initializer = tf.random_uniform_initializer(-0.01, 0.01)
		weights = self._get_variable(name='conv_weights', shape=kernel_shape, 
									 initializer=initializer)

		# Create variable named "biases".
		biases = self._get_variable(name='conv_bias', shape=bias_shape, 
									 initializer=initializer)
		conv = self.conv2d(x, weights)
		z = conv + biases
		bn = self.batch_normalization(z)
		relu = tf.nn.relu(bn)
		pool = self.max_pool_2x2(relu)
		return pool
	


	def batch_normalization(self, x):
		'''
		Realize batch normalization
		'''
		x_shape = x.get_shape()
		params_shape = x_shape[-1]
		axis = list(range(len(x_shape) - 1))
		#The bias for batch normalization
		beta = self._get_variable(name='beta',
							 shape=params_shape,
							 initializer=tf.zeros_initializer)
		#The scale of batch normalization
		gamma = self._get_variable(name='gamma',
							  shape=params_shape,
							  initializer=tf.ones_initializer)
		#Record moving average for testing
		moving_mean = self._get_variable(name='moving_mean',
									shape=params_shape,
									initializer=tf.zeros_initializer,
									trainable=False)
		#Record moving average for testing
		moving_variance = self._get_variable(name='moving_variance',
									shape=params_shape,
									initializer=tf.ones_initializer,
									trainable=False)  
		# These ops will only be preformed when training.
		mean, variance = tf.nn.moments(x, axis)
		#Update moving averages
		decay = 0.95
		update_moving_mean = tf.assign(moving_mean,
							   moving_mean * decay + mean * (1 - decay))
		update_moving_variance = tf.assign(moving_variance,
							  moving_variance * decay + variance * (1 - decay))

	  
		if self.is_training:
			x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
		else:
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				x = tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, 0.001) 
		
		return x

batch_size = 64
num_pixel = 784
num_class = 10
tf.reset_default_graph() 
graph = tf.Graph()

with graph.as_default():
	#Define Xavier initializer
	initializer = tf.random_uniform_initializer(-0.01, 0.01)
	with tf.variable_scope('CNN', initializer=initializer) as scope:
		input_holder = tf.placeholder(tf.float32, [None, num_pixel])
		#Note, the label is one-hot encoder
		target_holder = tf.placeholder(tf.float32, [None,num_class])
		train_model = CnnMnistModel(input_holder, target_holder, True, 0.5)
		tf.get_variable_scope().reuse_variables()
		test_model = CnnMnistModel(input_holder, target_holder, False, 1)


epochs = 1
batch_size = 64
num_steps = int(n_tr/batch_size)
print('n_tr =', n_tr, 'num_steps =', num_steps)
#Create a session
with tf.Session(graph=graph) as sess:
	#Initialize variables
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	#Train the datase
	fout = open('log_with_bn.txt', 'w')
	fout.close()
	for eeee in range(epochs):
		l, a = 0, 0
		fout = open('log_with_bn.txt', 'a')
		for step in range(num_steps):
			batch_data, batch_labels = mnist.train.next_batch(batch_size)
			feed_dict = {input_holder : batch_data, target_holder: batch_labels}
			#Train
			_, l, a = sess.run([train_model.optimize, train_model.loss, train_model.accuracy], 
							   feed_dict=feed_dict)
			if eeee == 0:
				s = str(l) + ' ' + str(a) + '\n'
				fout.write(s)
				print('step:', str(step), 'Loss:{:.5f}'.format(l), 'Accuracy:{:.5f}'.format(a))
		fout.close()
		print('epoch:', str(eeee), 'Loss:{:.5f}'.format(l), 'Accuracy:{:.5f}'.format(a))
	
	count = 0	
	for _ in range(200):
		batch_data, batch_labels = mnist.test.next_batch(50)
		feed_dict = {input_holder : batch_data, target_holder: batch_labels}
		cp = sess.run(test_model.correct_num, feed_dict=feed_dict)
		#print(cp)
		#feed_dict = {input_holder : batch_data, target_holder: batch_labels}
		#cp = sess.run(train_model.correct_num, feed_dict=feed_dict)
		#print(cp)
		count += np.sum(cp)
	print("Testing Accuracy：", count/n_ts)	# 0.9223

					