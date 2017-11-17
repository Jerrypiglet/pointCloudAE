# CUDA_VISIBLE_DEVICES=1 vglrun python train_ae_1_lmdb.py --task_name finalAe_FASTconstantLr_bnNObn_NOtrans_car24576__bb10 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=32
# vglrun python train_ae_1_lmdb.py --task_name finalAE_FASTconstantLr_bnNObn_NOtrans_car24576__bb10 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=10 --learning_rate=3e-6 --restore=True
# vglrun python train_ae_1_lmdb.py --task_name finalAE_1e-5_bnNObn_car24576__bb10 --num_point=24576 --if_constantLr=True --if_deconv=True --if_transform=False --if_en_bn=True --if_gen_bn=False --cat_name='car' --batch_size=10 --learning_rate=1e-5

import argparse
import math
import h5py
import numpy as np
# np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
# np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import socket
import importlib
import os
import sys
import time
import matplotlib.pyplot as plt
import scipy.io as sio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models/chen-hsuan'))
# import provider
import tf_util
import util
from pc_util import draw_point_cloud, point_cloud_three_views

# from utils_gen import *
from depthestimate import tf_nndistance
from pcd_ae_1_lmdb_24576_fullSlim import PCD_ae

np.random.seed(0)
tf.set_random_seed(0)
global FLAGS
flags = tf.flags
flags.DEFINE_integer('gpu', 0, "GPU to use [default: GPU 0]")
# task and control (yellow)
flags.DEFINE_string('model_file', 'pcd_ae_1_lmdb', 'Model name')
flags.DEFINE_string('cat_name', 'airplane', 'Category name')
flags.DEFINE_string('LOG_DIR', '/newfoundland/rz1/log/summary', 'Log dir [default: log]')
flags.DEFINE_string('CHECKPOINT_DIR', '/newfoundland/rz1/log', 'Log dir [default: log]')
flags.DEFINE_string('task_name', 'tmp', 'task name to create under /LOG_DIR/ [default: tmp]')
flags.DEFINE_boolean('restore', False, 'If resume from checkpoint')
# train (green)
flags.DEFINE_integer('num_point', 2048, 'Point Number [256/512/1024/2048] [default: 1024]')
flags.DEFINE_integer('batch_size', 32, 'Batch Size during training [default: 32]')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate [default: 0.001]') #used to be 3e-5
flags.DEFINE_float('momentum', 0.9, 'Initial learning rate [default: 0.9]')
flags.DEFINE_string('optimizer', 'adam', 'adam or momentum [default: adam]')
flags.DEFINE_integer('decay_step', 5000000, 'Decay step for lr decay [default: 200000]')
flags.DEFINE_float('decay_rate', 0.7, 'Decay rate for lr decay [default: 0.8]')
# arch (magenta)
flags.DEFINE_boolean('if_deconv', True, 'If add deconv output to generator aside from fc output')
flags.DEFINE_boolean('if_constantLr', True, 'If use constant lr instead of decaying one')
flags.DEFINE_boolean('if_en_bn', False, 'If use batch normalization for the mesh decoder')
flags.DEFINE_boolean('if_gen_bn', False, 'If use batch normalization for the mesh generator')
flags.DEFINE_float('bn_decay', 0.9, 'Decay rate for batch normalization [default: 0.9]')
flags.DEFINE_boolean("if_transform", False, "if use two transform layers")
flags.DEFINE_float('reg_weight', 0.1, 'Reweight for mat loss [default: 0.1]')
flags.DEFINE_boolean("if_vae", False, "if use VAE instead of vanilla AE")
flags.DEFINE_boolean("if_l2Reg", False, "if use l2 regularizor for the generator")
flags.DEFINE_float('vae_weight', 0.1, 'Reweight for mat loss [default: 0.1]')
# log and drawing (blue)
flags.DEFINE_boolean("if_summary", True, "if save summary")
flags.DEFINE_boolean("if_save", True, "if save")
flags.DEFINE_integer("save_every_step", 1000, "save every ? step")
flags.DEFINE_boolean("if_test", True, "if test")
flags.DEFINE_integer("test_every_step", 20, "test every ? step")
flags.DEFINE_boolean("if_draw", True, "if draw latent")
flags.DEFINE_integer("draw_every_step", 1000, "draw every ? step")
flags.DEFINE_boolean("if_init_i", False, "if init i from 0")
flags.DEFINE_integer("init_i_to", 1, "init i to")
FLAGS = flags.FLAGS

POINTCLOUDSIZE = FLAGS.num_point
if FLAGS.if_deconv:
	OUTPUTPOINTS = FLAGS.num_point
else:
	OUTPUTPOINTS = FLAGS.num_point/2
FLAGS.BN_INIT_DECAY = 0.5
FLAGS.BN_DECAY_DECAY_RATE = 0.5
FLAGS.BN_DECAY_DECAY_STEP = float(FLAGS.decay_step)
FLAGS.BN_DECAY_CLIP = 0.99

def log_string(out_str):
	FLAGS.LOG_FOUT.write(out_str+'\n')
	FLAGS.LOG_FOUT.flush()
	print(out_str)

def prepare_plot():
	# global figM_gt, ms_gt, figM_x, ms_x
	# figM_gt, ms_gt = prepare_for_showing3D(points_num=POINTCLOUDSIZE)
	# figM_x, ms_x = prepare_for_showing3D(points_num=OUTPUTPOINTS)
	
	pltfig_3d_gt = plt.figure(1, figsize=(25, 20))
	plt.axis('off')
	plt.show(block=False)

def save(ae, step, epoch, batch):
	# save_path = os.path.join(FLAGS.CHECKPOINT_DIR, FLAGS.task_name)
	saved_checkpoint = ae.saver.save(ae.sess, \
		FLAGS.CHECKPOINT_DIR + '/step%d-epoch%d-batch%d.ckpt' % (step, epoch, batch), \
		global_step=step)
	log_string(util.toBlue("-----> Model saved to file: %s; step = %d" % (saved_checkpoint, step)))

def restore(ae):
	restore_path = FLAGS.CHECKPOINT_DIR
	latest_checkpoint = tf.train.latest_checkpoint(restore_path)
	log_string(util.toYellow("-----> Model restoring from: %s..."%restore_path))
	ae.restorer.restore(ae.sess, latest_checkpoint)
	log_string(util.toYellow("----- Restored from %s."%latest_checkpoint))

def train(ae):
	num_samples = ae.gen.x_size_train
	batch_size = ae.batch_size

	if FLAGS.if_init_i:
		i = FLAGS.init_i_to
		log_string(util.toGreen("======== init i to %s"%FLAGS.init_i_to))
	else:
		i = ae.sess.run(ae.global_i)
		log_string(util.toGreen("======== restore i to %s"%i))

	def write_to_screen(merged, loss, chamfer_loss, mat_loss, latent_loss, step, start_time, is_training=True, input_data=None, x_recon=None):
		if FLAGS.if_summary:
			ae.train_writer.add_summary(merged, step)
			ae.train_writer.flush()

		epoch_show = math.floor(float(step) * batch_size / float(num_samples))
		batch_show = math.floor(step - epoch_show * (num_samples / batch_size))
		if not(is_training):
			log_string(util.toGreen('--Testing...--'))

		if FLAGS.if_save and i != 0 and step % FLAGS.save_every_step == 0 and is_training:
			save(ae, step, epoch_show, batch_show)

		end_time = time.time()
		elapsed = end_time - start_time

		out_string = "i %03d Epo %03d ba %03d loss %.4f: chamfer %.4f + reg %.4f + vae %.4f. -Time %f sec." % (i, epoch_show, batch_show, \
			loss, chamfer_loss, mat_loss, latent_loss, elapsed)
		log_string(out_string)
		if not(is_training):
			log_string(util.toGreen('--Done.--'))
			if FLAGS.if_draw and step % FLAGS.draw_every_step == 0:
				plot_indexs = [[421, 423], [422, 424], [425, 427], [426, 428]]
				fig = plt.figure(1)
				plt.clf()
				for idx, plot_index in enumerate(plot_indexs):
					plt.subplot(plot_index[0])
					plt.imshow(point_cloud_three_views(input_data[idx]))
					plt.axis('off')
					plt.subplot(plot_index[1])
					plt.imshow(point_cloud_three_views(x_recon[idx]))
					plt.axis('off')
				fig.suptitle(FLAGS.task_name)
				fig.canvas.draw()
				plt.pause(0.001)
				save_images_folder = os.path.join(FLAGS.LOG_DIR, 'saved_images')
				# save_images_folder = '/home/rz1/Dropbox/optim_plots/saved_images'
				png_name = os.path.join(save_images_folder, 'step%d_%d-epoch%d-batch%d.png'%(i, idx, epoch_show, batch_show))
				fig.savefig(png_name)
				log_string(util.toBlue('>> Vis saved at: '+png_name))

	try:
		while not ae.coord.should_stop():
			ae.sess.run(ae.assign_i_op, feed_dict={ae.set_i_to_pl: i})
			start_time = time.time()
			feed_dict = {ae.is_training_pl: True, ae.gen.is_training_pl: True}
			summary, step, _, loss, chamfer_loss, mat_loss, latent_loss,x_recon = ae.sess.run([ae.merged_train, ae.counter,
				ae.train_op, ae.loss, ae.chamfer_loss, ae.mat_diff_loss, ae.latent_loss, ae.x_recon], feed_dict=feed_dict)
			write_to_screen(summary, loss, chamfer_loss, mat_loss, latent_loss, i, start_time, is_training=True)

			if ((FLAGS.if_test and i % FLAGS.test_every_step == 0) or (FLAGS.if_draw and i % FLAGS.draw_every_step == 0)) and i != 0:
				start_time = time.time()
				feed_dict = {ae.is_training_pl: False, ae.gen.is_training_pl: False}
				if FLAGS.if_vae:
					summary, _, _, loss, chamfer_loss, mat_loss, latent_loss, x_val, x_recon_val, z_mean, z_log_sigma_sq, z_std = \
						ae.sess.run([ae.merged_test, ae.counter,
						ae.test_count_op, ae.loss, ae.chamfer_loss, ae.mat_diff_loss, ae.latent_loss, ae.point_cloud, ae.x_recon, ae.z_mean, ae.z_log_sigma_sq, ae.z_std], feed_dict=feed_dict)
					print z_mean[:2], z_mean.shape, np.amax(z_mean), np.amin(z_mean)
					print z_log_sigma_sq[:2], z_log_sigma_sq.shape, np.amax(z_log_sigma_sq), np.amin(z_log_sigma_sq)
					print z_std[:2], z_std.shape, np.amax(z_std), np.amin(z_std)
				else:
					summary, _, _, loss, chamfer_loss, mat_loss, latent_loss, x_val, x_recon_val = \
						ae.sess.run([ae.merged_test, ae.counter,
						ae.test_count_op, ae.loss, ae.chamfer_loss, ae.mat_diff_loss, ae.latent_loss, ae.point_cloud, ae.x_recon], feed_dict=feed_dict)
				write_to_screen(summary, loss, chamfer_loss, mat_loss, latent_loss, i, start_time, is_training=False, input_data=x_val, x_recon=x_recon_val)

			i = i + 1
	except tf.errors.OutOfRangeError:
		print('Done training.')
	finally:
		# When done, ask the threads to stop.
		ae.coord.request_stop()
	# Wait for threads to finish.
	ae.coord.join(ae.threads)
	ae.sess.close()

def test_demo_render_z(ae, z_list = []):
	# scp -r jerrypiglet@128.237.196.156:/Users/jerrypiglet/Bitsync/3dv2017_PBA/models . && scp -r jerrypiglet@128.237.196.156:/Users/jerrypiglet/Bitsync/3dv2017_PBA/train_ae_1_lmdb.py . && CUDA_VISIBLE_DEVICES=0 vglrun python train_ae_1_lmdb.py --task_name ae_deconv_constantLr_NObn_NOtrans_car8192_fullSlim__bb10 --num_point=8192 --if_constantLr=True --if_deconv=True --if_transform=False --if_gen_bn=False --cat_name='car' --if_vae=False --batch_size=32 --restore=True --if_save=False --if_summary=False
	num_samples = ae.gen.x_size_train
	batch_size = ae.batch_size
	i = 0
	
	try:
		write_folder = './z_mean.mat'
		while not ae.coord.should_stop():
			start_time = time.time()
			feed_dict = {ae.is_training_pl: True, ae.gen.is_training_pl: True}
			z_mean = \
				ae.sess.run(ae.feat, feed_dict=feed_dict)
			z_list.append(z_mean)
			i = i + 1
			print i, ae.gen.x_size_train, ae.batch_size

			if i>=ae.gen.x_size_train // ae.batch_size:
				print i
				z_all = np.concatenate(z_list)
				sio.savemat(write_folder,{'z_all': z_all})
				print write_folder+' Saved.'

				std_all = np.std(z_all, axis=0)
				max_index = np.flip(np.argsort(std_all), 0)
				std_max = std_all[max_index]
				print std_max[:20]
				dims = max_index[:20]
				print dims.tolist(), len(dims.tolist())

				# w, v = np.linalg.eig(np.dot(np.transpose(z_all), z_all))
				
				steps = 30

				# for dim in range(0, 200):
				for dim_index in range(20):
					# dim = 1
					# eig_value = w[dim]
					# eig_vec = v[:, dim:dim+1]
					# z_one = np.reshape(np.linspace(-3.*eig_value, 3.*eig_value, steps), (steps, 1))
					# print z_one
					# z = np.dot(z_one, np.transpose(eig_vec))

					dim = dims[dim_index]
					z_one = np.reshape(np.linspace(np.mean(z_all[:, dim])-3.*np.std(z_all[:, dim]), np.mean(z_all[:, dim])+3.*np.std(z_all[:, dim]), steps), (steps, 1))
					z = np.zeros((steps, 20))
					z[:, dim:dim+1] = z_one
					feed_dict={ae.is_training_pl: True, ae.gen.is_training_pl: True, ae.feat: z}

					x_recon_val = ae.sess.run(ae.x_recon, feed_dict=feed_dict)
					
					for step in range(steps):
						fig = plt.figure(10, figsize=(15, 7))
						plt.clf()
						plt.imshow(point_cloud_three_views(x_recon_val[step]))
						plt.axis('off')
						fig.suptitle('eigen %d'%(dim_index))
						fig.canvas.draw()
						plt.pause(0.001)
						png_name = './saved_images/dim%d-step%d.png'%(dim_index, step)
						fig.savefig(png_name)
						log_string(util.toBlue('>> Vis saved at'+png_name))

	except tf.errors.OutOfRangeError:
		print('Done test_demo_render_z.')
	finally:
		# When done, ask the threads to stop.
		ae.coord.request_stop()
	# Wait for threads to finish.
	ae.coord.join(ae.threads)
	ae.sess.close()
		 
if __name__ == "__main__":
	MODEL = importlib.import_module(FLAGS.model_file) # import network module
	MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model_file+'.py')
	
	FLAGS.LOG_DIR = FLAGS.LOG_DIR + '/' + FLAGS.task_name
	FLAGS.CHECKPOINT_DIR = os.path.join(FLAGS.CHECKPOINT_DIR, FLAGS.task_name)
	tf_util.mkdir(FLAGS.CHECKPOINT_DIR)
	if not os.path.exists(FLAGS.LOG_DIR):
		os.mkdir(FLAGS.LOG_DIR)
		print util.toYellow('===== Created %s.'%FLAGS.LOG_DIR)
	else:
		# os.system('rm -rf %s/*'%FLAGS.LOG_DIR)
		if not(FLAGS.restore):
			delete_key = raw_input(util.toRed('===== %s exists. Delete? [y (or enter)/N] '%FLAGS.LOG_DIR))
			if delete_key == 'y' or delete_key == "":
				os.system('rm -rf %s/*'%FLAGS.LOG_DIR)
				os.system('rm -rf %s/*'%FLAGS.CHECKPOINT_DIR)
				print util.toRed('Deleted.'+FLAGS.LOG_DIR+FLAGS.CHECKPOINT_DIR)
			else:
				print util.toRed('Overwrite.')
		else:
			print util.toRed('To Be Restored...')

	tf_util.mkdir(os.path.join(FLAGS.LOG_DIR, 'saved_images'))
	os.system('cp %s %s' % (MODEL_FILE, FLAGS.LOG_DIR)) # bkp of model def
	os.system('cp train.py %s' % (FLAGS.LOG_DIR)) # bkp of train procedure


	FLAGS.LOG_FOUT = open(os.path.join(FLAGS.LOG_DIR, 'log_train.txt'), 'w')
	FLAGS.LOG_FOUT.write(str(FLAGS)+'\n')

	prepare_plot()
	log_string(util.toYellow('<<<<'+FLAGS.task_name+'>>>> '+str(tf.flags.FLAGS.__flags)))

	ae = PCD_ae(FLAGS)
	if FLAGS.restore:
		restore(ae)
	train(ae)

	# z_list = []
	# test_demo_render_z(ae, z_list)

	FLAGS.LOG_FOUT.close()