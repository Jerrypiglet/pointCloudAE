import tensorflow as tf
import tflearn
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../data'))
import tf_util
import util
from transform_nets import input_transform_net, feature_transform_net
from depthestimate import tf_nndistance
import tensorflow.contrib.layers as ly
import tensorflow.contrib.slim as slim
from pcd_lmdb_loader import PCD_loader

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * abs(x)

class PCD_ae(object):
	def __init__(self, FLAGS):
		self.FLAGS = FLAGS
		self.activation_fn = lrelu
		# self.activation_fn = tf.nn.relu
		self.batch_size = self.FLAGS.batch_size
		self.num_point = self.FLAGS.num_point
		self.is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training_pl')

		# Note the global_step=batch parameter to minimize. 
		# That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
		self.counter = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
		# self.bn_decay = get_bn_decay(self.counter, self.FLAGS)
		self.test_count = tf.Variable(0, trainable=False)
		self.test_count_op = tf.assign(self.test_count, tf.add(self.test_count, tf.constant(1)))

		self.global_i = tf.Variable(0, name='global_i', trainable=False)
		self.set_i_to_pl = tf.placeholder(tf.int32,shape=[], name='set_i_to_pl')
		self.assign_i_op = tf.assign(self.global_i, self.set_i_to_pl)

		self.gen = PCD_loader(self.FLAGS)

		self._log_string(util.toGreen('-----> Defining network...'))
		self._create_network()
		self._log_string(util.toGreen('-----> Defining loss, optimizer and summary...'))
		self._create_loss()
		self._create_optimizer()
		self._create_summary()

		self._print_arch(str([var.name for var in tf.trainable_variables()]))

		# Add ops to save and restore all the variables.
		self.saver = tf.train.Saver(max_to_keep=2)
		self.restorer = tf.train.Saver()
			
		# Create a session
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())

		# # Start input enqueue threads.
		self.coord = tf.train.Coordinator()
		self._log_string(util.toGreen("===== main-->tf.train.start_queue_runners"))
		self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

		self.train_writer = tf.summary.FileWriter(os.path.join(FLAGS.LOG_DIR, 'train'), self.sess.graph)

	def _log_string(self, out_str):
		self.FLAGS.LOG_FOUT.write(out_str+'\n')
		self.FLAGS.LOG_FOUT.flush()
		print(out_str)

	def _print_arch(self, out_str, tensor=None):
		if tensor==None:
			self._log_string(util.toMagenta(out_str))
		else:
			self._log_string(util.toMagenta(out_str + str(tensor.get_shape().as_list())))

	def _create_encoder(self, input_sample, trainable=True, if_bn=False, reuse=False, scope_name='encoder'):
		 with tf.variable_scope(scope_name) as scope:
			if reuse:
				scope.reuse_variables()

			if if_bn:
				self._print_arch('=== Using BN for ENCODER!')
				batch_normalizer_en = slim.batch_norm
				batch_norm_params_en = {'is_training': self.is_training_pl, 'decay': self.FLAGS.bn_decay, 'updates_collections': None}
			else:
				self._print_arch('=== NOT Using BN for ENCODER!')
				batch_normalizer_en = None
				batch_norm_params_en = None

			with slim.arg_scope([slim.fully_connected, slim.conv2d], 
					activation_fn=self.activation_fn,
					trainable=trainable,
					normalizer_fn=batch_normalizer_en,
					normalizer_params=batch_norm_params_en):
				with tf.device('/gpu:0'):
					net = slim.conv2d(self.input_image, 32, kernel_size=[1,3], stride=[1,1], padding='VALID',scope='conv1')
					self._print_arch("++ After 'slim.conv1': ", net) #[32, N, 1, 64]
					net = slim.conv2d(net, 64, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv2')
					self._print_arch("++ After 'slim.conv2': ", net) #[32, N, 1, 64]

					if self.FLAGS.if_transform:
						with tf.variable_scope('transform_self.net2') as sc:
								transform = feature_transform_net(net, self.is_training_pl, self.bn_decay, K=64)
						self.end_points_transform = transform
						net_transformed = tf.matmul(tf.squeeze(net), transform)
						self._print_arch("++ After 'transform_self.net2': ", net_transformed) #[32, N, 64]
						net = tf.expand_dims(net_transformed, [2])
						self._print_arch("++ After 'transform_self.net2': ", net) #[32, N, 1, 64]

					net = slim.conv2d(net, 64, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv3')
					self._print_arch("++ After 'slim.conv3': ", net) #[32, N, 1, 64]

				with tf.device('/gpu:1'):
					net = slim.conv2d(net, 128, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv4')
					self._print_arch("++ After 'slim.conv4': ", net) #[32, N, 1, 128]
					net = slim.conv2d(net, 256, kernel_size=[1,1], stride=[1,1], padding='VALID',scope='conv5')
					self._print_arch("++ After 'slim.conv5': ", net) #[32, N, 1, 1024]
					feat = slim.max_pool2d(net, [self.num_point,1], padding='VALID', scope='maxpool')
					self._print_arch("++ After 'slim.max_pool2d': ", feat) #[32, 1, 1, 1024]

					self.feat_before_VAE = tf.reshape(feat, [-1, 256]) #[32, 1024]

					if self.FLAGS.if_vae:
						self._print_arch('=== Using VAE!')
						with slim.arg_scope([slim.fully_connected],
								trainable=trainable):
							batch_normalizer_vae = slim.batch_norm
							batch_norm_params_vae = {'is_training': self.is_training_pl, 'decay': self.FLAGS.bn_decay, 'updates_collections': None}

							z_mean = slim.fully_connected(self.feat_before_VAE, 1024, activation_fn=self.activation_fn, normalizer_fn=batch_normalizer_vae, normalizer_params=batch_norm_params_vae, scope='vae_mean1')
							z_mean = slim.fully_connected(z_mean, 1024, activation_fn=self.activation_fn, normalizer_fn=None, normalizer_params=None, scope='vae_mean2')
							self.z_mean = slim.fully_connected(z_mean, 1024, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='vae_mean3')
							
							z_log_sigma_sq = slim.fully_connected(self.feat_before_VAE, 1024, activation_fn=self.activation_fn, normalizer_fn=batch_normalizer_vae, normalizer_params=batch_norm_params_vae, scope='vae_var1')
							z_log_sigma_sq = slim.fully_connected(z_log_sigma_sq, 1024, activation_fn=self.activation_fn, normalizer_fn=None, normalizer_params=None, scope='vae_var2')
							self.z_log_sigma_sq = slim.fully_connected(z_log_sigma_sq, 1024, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='vae_var3')
							self.z_log_sigma_sq = tf.clip_by_value(self.z_log_sigma_sq, -50., 10.)

						eps = tf.random_normal(tf.stack([self.dyn_batch_size_x, 1024]))
						self.z_std = tf.sqrt(tf.exp(self.z_log_sigma_sq))
						# self.feat = self.z_mean + eps * self.z_std
						self.feat = tf.add(self.z_mean, tf.multiply(self.z_std, eps))
					else:
						self.feat = self.feat_before_VAE

			return self.feat

	def _create_generator(self, feat, trainable=True, if_bn=False, reuse=False, scope_name='generator'):
		 with tf.variable_scope(scope_name) as scope:
			if reuse:
				scope.reuse_variables()

			if if_bn:
				self._print_arch('=== Using BN for GENERATOR!')
				batch_normalizer_gen = slim.batch_norm
				batch_norm_params_gen = {'is_training': self.is_training_pl, 'decay': self.FLAGS.bn_decay, 'updates_collections': None}
			else:
				self._print_arch('=== NOT Using BN for GENERATOR!')
				batch_normalizer_gen = None
				batch_norm_params_gen = None

			if self.FLAGS.if_l2Reg:
				self._print_arch('=== Using L2 regularizor for GENERATOR!')
				weights_regularizer = slim.l2_regularizer(1e-5)
			else:
				weights_regularizer = None

			with tf.device('/gpu:2'):
				with slim.arg_scope([slim.fully_connected], 
						activation_fn=self.activation_fn,
						trainable=trainable,
						normalizer_fn=batch_normalizer_gen,
						normalizer_params=batch_norm_params_gen, 
						weights_regularizer=weights_regularizer):
					x_additional = slim.fully_connected(self.feat, 2048, scope='gen_fc1')
					x_additional = slim.fully_connected(x_additional, 4096, scope='gen_fc2')
					x_additional = slim.fully_connected(x_additional, 8192, scope='gen_fc3')
					x_additional = slim.fully_connected(x_additional, 8192*3, scope='gen_fc4',
						activation_fn=None, normalizer_fn=None, normalizer_params=None)
				
				self.x_recon=tf.reshape(x_additional,(-1,8192,3))

			with tf.device('/gpu:3'):
				if self.FLAGS.if_deconv:
					x_additional_conv = tf.reshape(slim.fully_connected(self.feat, 1024, activation_fn=self.activation_fn), [-1, 4, 4, 64])
					with slim.arg_scope([slim.conv2d_transpose], 
							activation_fn=self.activation_fn,
							trainable=trainable,
							normalizer_fn=batch_normalizer_gen,
							normalizer_params=batch_norm_params_gen, 
							weights_regularizer=weights_regularizer):
						gen_deconv1 = slim.conv2d_transpose(x_additional_conv, 256, kernel_size=[3,3], stride=[1,1], padding='VALID',scope='gen_deconv1')
						gen_deconv2 = slim.conv2d_transpose(gen_deconv1, 128, kernel_size=[3,3], stride=[1,1], padding='VALID',scope='gen_deconv2')
						gen_deconv3 = slim.conv2d_transpose(gen_deconv2, 64, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv3')
						gen_deconv4 = slim.conv2d_transpose(gen_deconv3, 64, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv4')
						gen_deconv5 = slim.conv2d_transpose(gen_deconv4, 32, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv6')
						gen_deconv6 = slim.conv2d_transpose(gen_deconv5, 3, kernel_size=[5,5], stride=[2,2], padding='SAME',scope='gen_deconv7',
							activation_fn=None, normalizer_fn=None, normalizer_params=None)
						self._print_arch("++ After 'deconv1': ", gen_deconv1)
						self._print_arch("++ After 'deconv2': ", gen_deconv2)
						self._print_arch("++ After 'deconv3': ", gen_deconv3)
						self._print_arch("++ After 'deconv4': ", gen_deconv4)
						self._print_arch("++ After 'deconv5': ", gen_deconv5)
						self._print_arch("++ After 'deconv6': ", gen_deconv6)

					self.x_recon_conv = tf.reshape(gen_deconv6, [-1, 128*128, 3])
					self._print_arch("++ Decov output shape shape: ", self.x_recon_conv)

					self.x_recon = tf.concat([self.x_recon, self.x_recon_conv],1)
					self._print_arch("++ Final concat shape: ", self.x_recon)

	def _create_network(self):
		## Define model

		with tf.device('/gpu:0'):
			self.point_cloud = self.gen.point_cloud_batch
			self.dyn_batch_size_x = tf.shape(self.point_cloud)[0]
			self._print_arch("++ Input: ", self.point_cloud) #[32, 1024, 3]

			if self.FLAGS.if_transform:
				with tf.variable_scope('transform_net1') as sc:
					transform = input_transform_net(self.point_cloud, self.is_training_pl, self.bn_decay, K=3)
				self.point_cloud_tran3 = tf.matmul(self.point_cloud, transform)
				self.input_image = tf.expand_dims(self.point_cloud_tran3, -1)
				self._print_arch("++ After 'transform_net1': ", self.input_image) #[32, 1024, 3]
			else:
				self.input_image = tf.expand_dims(self.point_cloud, -1)
				self._print_arch("++ After 'tf.expand_dims': ", self.input_image) #[32, 1024, 3,1]

		## === shape encoder + vae ===
		self._create_encoder(self.input_image, trainable=True, if_bn=self.FLAGS.if_en_bn, reuse=False, scope_name='encoder')

		with tf.device('/gpu:1'):
			## === shape generator ===
			self._create_generator(self.feat, trainable=True, if_bn=self.FLAGS.if_gen_bn, reuse=False, scope_name='generator')
		
	def _create_loss(self):
			""" pred: B*NUM_CLASSES,
					label: B, """
			# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
			# chamfer_loss = tf.reduce_mean(loss)

			dists_forward,_,dists_backward,_=tf_nndistance.nn_distance(self.point_cloud, self.x_recon)
			mindist=dists_forward
			dist0=mindist[0,:]
			dists_forward=tf.reduce_mean(dists_forward)
			dists_backward=tf.reduce_mean(dists_backward)
			loss_nodecay=(dists_forward+dists_backward/2.0)*10000
			self.chamfer_loss=loss_nodecay
				# +tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*0.1

			if self.FLAGS.if_transform:
				# Enforce the transformation as orthogonal matrix
				transform = self.end_points_transform # BxKxK
				K = transform.get_shape()[1].value
				mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
				mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
				self.mat_diff_loss = tf.nn.l2_loss(mat_diff)
			else:
				self.mat_diff_loss = tf.constant(0.)

			if self.FLAGS.if_vae:
				self.latent_loss = tf.reduce_mean(
					tf.reduce_sum(
						0.5 * (tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq) - self.z_log_sigma_sq - 1.0)
						, 1)
					)
			else:
				self.latent_loss = tf.constant(0.)

			self.loss = self.chamfer_loss + self.mat_diff_loss * self.FLAGS.reg_weight + self.latent_loss * self.FLAGS.vae_weight

	def _create_optimizer(self):
		# Get training operator
		if self.FLAGS.if_constantLr:
			self.learning_rate = self.FLAGS.learning_rate
			self._log_string(util.toGreen('===== Using constant lr!'))
		else:  
			self.learning_rate = get_learning_rate(self.counter, self.FLAGS)
		if self.FLAGS.optimizer == 'momentum':
			self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=FLAGS.momentum)
		elif self.FLAGS.optimizer == 'adam':
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss, global_step=self.counter)

	def _create_summary(self):
		# self.summary_bn_decay = tf.summary.scalar('train/bn_decay', self.bn_decay)
		self.summary_learning_rate = tf.summary.scalar('train/learning_rate', self.learning_rate)

		self.summary_loss_train = tf.summary.scalar('train/loss', self.loss)
		self.summary_loss_test = tf.summary.scalar('test/loss', self.loss)
		self.summary_chamfer_loss_train = tf.summary.scalar('train/chamfer loss', self.chamfer_loss)
		self.summary_chamfer_loss_test = tf.summary.scalar('test/chamfer loss', self.chamfer_loss)
		self.summary_mat_loss_train = tf.summary.scalar('train/mat loss', self.mat_diff_loss)
		self.summary_mat_loss_test = tf.summary.scalar('test/mat loss', self.mat_diff_loss)
		self.summary_feat_hist_test = tf.summary.histogram("test/feat", self.feat)

		self.summary_train = [self.summary_chamfer_loss_train, self.summary_mat_loss_train]
		self.summary_test = [self.summary_chamfer_loss_test, self.summary_mat_loss_test, self.summary_feat_hist_test]

		if self.FLAGS.if_vae:
			self.summary_feat_before_VAE_hist_test = tf.summary.histogram("test/feat_before_VAE", self.feat_before_VAE)
			self.summary_test += [self.summary_feat_before_VAE_hist_test]
		
		if self.FLAGS.if_vae:
			self.summary_vae_loss_train = tf.summary.scalar('train/vae loss', self.latent_loss)
			self.summary_vae_loss_test = tf.summary.scalar('test/vae loss', self.latent_loss)
			self.summary_z_mean_hist_test = tf.summary.histogram("test/z_mean", self.z_mean)
			self.summary_z_std_hist_test = tf.summary.histogram("test/z_std", self.z_std)
			self.summary_test += [self.summary_vae_loss_train, self.summary_vae_loss_test, self.summary_z_mean_hist_test, self.summary_z_std_hist_test]

		self.merge_list_train = [self.summary_loss_train, self.summary_learning_rate] + self.summary_train
		self.merge_list_test = [self.summary_loss_test] + self.summary_test
		self.merged_train = tf.summary.merge(self.merge_list_train)
		self.merged_test = tf.summary.merge(self.merge_list_test)


if __name__=='__main__':
		with tf.Graph().as_default():
				inputs = tf.zeros((32,1024,3))
				outputs = get_model(inputs, tf.constant(True))
	# print outputs
