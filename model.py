import tensorflow as tf
import numpy as np
import glob
import random
import utils
import os
import imageio as imio

# lambda for convolution function
conv = lambda inp, filt, str, pad, id : tf.nn.conv2d(input   = inp,
													 filter  = filt,
													 strides = [1, str, str, 1],
													 padding = pad,
													 name    = id)

# lambda for deconvolution function
deconv = lambda inp, filt, st, pad, name, op_shape : tf.nn.conv2d_transpose(value = inp,
														   	                filter = filt,
													                        output_shape = op_shape,
													                        strides = [1, st, st, 1],
													                        padding = pad,
													                        name = name)

class dip:
	def __init__(self, sess, fname, k1=5, k2=3, k3=3, l1=50,
		l2=50, l3=50, batch_size=1, epochs=500, save_freq=100,
		sample_freq=100, inpaint=False, denoise=True):
		self.inpaint = inpaint
		self.denoise = denoise
		self.img = imio.imread(fname).astype(np.float32)
		self.original = self.img/255.
		self.mask = np.ones_like(self.img)
		if self.inpaint:
			self.mask[120:190, 140:200] = 0.
		self.img = self.img*self.mask
		if self.denoise:
			self.img += np.random.normal(0, 25., self.img.shape)
			self.img = np.clip(self.img, 0, 255)
		imio.imwrite('input_image.png', self.img.astype(np.uint8))
		self.img = self.img/255.
		self.a1, self.a2, _ = self.img.shape
		self.img = self.img.reshape(1, self.a1, self.a2, 3)
		self.mask = self.mask.reshape(1, self.a1, self.a2, 3)

		self.K1 = k1
		self.K2 = k2
		self.K3 = k3
		self.L1 = l1
		self.L2 = l2
		self.L3 = l3
		self.save_freq = save_freq
		self.sample_freq = sample_freq

		self.batch_size = batch_size
		self.epochs = epochs
		self.sess = sess
		self.model_dir = './checkpoint'
		self.test_dir = './test'
		self.create_model()

	def load(self):
		checkpoint = tf.train.get_checkpoint_state(self.model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.inp_noise = np.load('checkpoint/inp_noise.npy')
			checkpt_name = os.path.basename(checkpoint.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.model_dir, checkpt_name))
			return True
		else:
			return False

	def create_model(self):
		self.img_train  = tf.placeholder(tf.float32, shape=[1, self.a1, self.a2, 3], name='img_input')
		self.img_label  = tf.placeholder(tf.float32, shape=[1, self.a1, self.a2, 3], name="img_label")

		print "input image", self.img_label

		self.w1         = tf.Variable(tf.truncated_normal(
								shape=[self.K1, self.K1, 3, self.L1],
								stddev=0.01),
							name='w1')
		self.conv1      = tf.nn.relu(conv(self.img_train, self.w1, 1, 'VALID', 'conv1'))
		print "conv1", self.conv1
		self.conv1_     = tf.contrib.layers.batch_norm(self.conv1)

		self.w2         = tf.Variable(tf.truncated_normal(
								shape=[self.K2, self.K2, self.L1, self.L2],
								stddev=0.01),
							name='w2')
		self.conv2      = tf.nn.relu(conv(self.conv1_, self.w2, 2, 'VALID', 'conv2'))
		print "conv2", self.conv2
		self.conv2_     = tf.contrib.layers.batch_norm(self.conv2)

		self.w3         = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L2, self.L3],
								stddev=0.01),
							name='w3')
		self.conv3      = tf.nn.relu(conv(self.conv2_, self.w3, 2, 'VALID', 'conv3'))
		print "conv3", self.conv3
		self.conv3_     = tf.contrib.layers.batch_norm(self.conv3)

		self.w4         = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L3, self.L3],
								stddev=0.01),
							name='w4')
		self.conv4      = tf.nn.relu(conv(self.conv3_, self.w4, 2, 'VALID', 'conv4'))
		print "conv4", self.conv4
		self.conv4_     = tf.contrib.layers.batch_norm(self.conv4)

		self.w5         = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L3, self.L3],
								stddev=0.01),
							name='w5')
		self.conv5      = tf.nn.relu(conv(self.conv4_, self.w5, 1, 'SAME', 'conv5'))
		print "conv5", self.conv5
		self.conv5_     = tf.contrib.layers.batch_norm(self.conv5)

		self.w6         = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L3, self.L3],
								stddev=0.01),
							name='w6')
		self.conv6      = tf.nn.relu(conv(self.conv5_, self.w6, 1, 'SAME', 'conv6'))
		print "conv6", self.conv6
		self.conv6_     = tf.contrib.layers.batch_norm(self.conv6)

		self.w7         = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L3, self.L3],
								stddev=0.01),
							name='w7')
		self.conv7      = tf.nn.relu(conv(self.conv6_, self.w6, 1, 'SAME', 'conv7'))
		print "conv7", self.conv7
		self.conv7_     = tf.contrib.layers.batch_norm(self.conv7)

		self.dw1        = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L3, self.L3],
								stddev=0.01),
							name='dw1')
		self.deconv1   = tf.nn.relu(deconv(self.conv7_, self.dw1, 2, 'VALID', 'deconv1', tf.shape(self.conv3_)))
		print "deconv1", self.deconv1

		self.dw2        = tf.Variable(tf.truncated_normal(
								shape=[self.K3, self.K3, self.L2, self.L3],
								stddev=0.01),
							name='dw2')
		self.deconv2   = tf.nn.relu(deconv(self.deconv1, self.dw2, 2, 'VALID', 'deconv2', tf.shape(self.conv2_)))

		# for adding skip connections
		# self.deconv2_   = tf.nn.relu(deconv(self.deconv1, self.dw2, 1, 'VALID', 'deconv1', tf.shape(self.conv2_)))
		# self.deconv2    = tf.concat([self.deconv2_, self.conv2], 3)
		print "deconv2", self.deconv2

		self.dw3        = tf.Variable(tf.truncated_normal(
								shape=[self.K2, self.K2, self.L1, self.L2],
								stddev=0.01),
							name='dw3')
		self.deconv3   = tf.nn.relu(deconv(self.deconv2, self.dw3, 2, 'VALID', 'deconv3', tf.shape(self.conv1_)))
		
		# for adding skip connections
		# self.dw3        = tf.Variable(tf.truncated_normal(
		# 						shape=[self.K2, self.K2, self.L1, self.L2+self.L2],
		# 						stddev=0.01),
		# 					name='dw3')
		# self.deconv3_   = tf.nn.relu(deconv(self.deconv2, self.dw3, 1, 'VALID', 'deconv3', tf.shape(self.conv1_)))
		# self.deconv3    = tf.concat([self.deconv3_, self.conv1], 3)
		print "deconv3", self.deconv3

		self.dw4        = tf.Variable(tf.truncated_normal(
								shape=[self.K1, self.K1, 3, self.L1],
								stddev=0.01),
							name='dw4')
		
		# for adding skip connection
		# self.dw4        = tf.Variable(tf.truncated_normal(
		# 						shape=[self.K1, self.K1, 3, self.L1+self.L1],
		# 						stddev=0.01),
		# 					name='dw4')
		self.deconv4    = tf.sigmoid(deconv(self.deconv3, self.dw4, 1, 'VALID', 'deconv4', tf.shape(self.img_label)))
		print "deconv4", self.deconv4

		self.L1 = 0.
		self.L2 = 1.

		self.error = tf.reduce_mean(self.mask*(self.deconv4 - self.img_label)**2)

		self.optim = tf.train.AdamOptimizer().minimize(self.error)

		self.saver = tf.train.Saver(max_to_keep=1)

		self.error_graph = tf.summary.scalar("Error", self.error)

	def init_var(self):
		init_op = tf.global_variables_initializer()
		self.sess.run(init_op)
		self.inp_noise = np.random.normal(0, 1., self.img.shape)

	def train_model(self):
		self.init_var()

		if self.load():
			print "[*] Model loaded successfully"
		else:
			print "[!] Model could not be loaded"

		self.writer = tf.summary.FileWriter('./logs', self.sess.graph)

		step_count = 1
		np.save('checkpoint/inp_noise', self.inp_noise)
		for e in range(self.epochs):
			_, er, graph_ = self.sess.run([self.optim, self.error, self.error_graph],
											feed_dict={self.img_train: self.inp_noise, self.img_label: self.img})

			self.writer.add_summary(graph_, step_count)

			print "Epochs {}/{} Error {}".format(e, self.epochs, er)

			if ((e+1) % self.save_freq == 0):
				self.saver.save(self.sess, self.model_dir+'/denoise', global_step=e)
			
			if (e % self.sample_freq) == 0:
				if not os.path.exists('./samples'):
					os.makedirs('./samples')
				output = self.sess.run(self.deconv4,
										feed_dict={self.img_train: self.inp_noise, self.img_label: self.img})
				if self.denoise:
					mse = np.mean((output[0]-self.original)**2)
					psnr = 20.*np.log10(1./np.sqrt(mse))
					print "----------------------------------------"
					print "Sample PSNR {}".format(psnr)
					print "----------------------------------------"
				imio.imwrite('samples/sample_{}.png'.format(e), (output[0]*255.).astype(np.uint8))

	# generate output for random noise centered around
	# the noise used for training the model
	def sample_for_noises(self):
		self.train_model()
		noise_samples = 100 # number of samples to be generated
		inps = np.random.normal(self.inp_noise,
								1.0,
								[noise_samples, 1, self.inp_noise.shape[1], self.inp_noise.shape[2], 3])

		if not os.path.exists('./samples_from_noises'):
			os.makedirs('samples_from_noises')

		for i in range(noise_samples):
			inp = inps[i].reshape(self.inp_noise.shape)
			output = self.sess.run(self.deconv4,
									feed_dict={self.img_train: inp, self.img_label: self.img})
			imio.imwrite('samples_from_noises/sample_{}.png'.format(i), (output[0]*255.).astype(np.uint8))