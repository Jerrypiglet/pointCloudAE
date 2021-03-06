import numpy as np
import math
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
from tensorpack import *
import gc
from plyfile import PlyData, PlyElement
import tables
from multiprocessing import Pool
import os
# from contextlib import closingy

sample_num = 24576

categories = [
	# "02691156", #airplane
	# "02690373" #airliner
        # "02828884",
	# "02933112",
	# "03001627", #chair
	# "03211117",
	# "03636649",
	# "03691459",
	# "04090263",
	# "04256520", #sofa
	# "04379243", 
	# "04401088",
	# "04530566",
	"02958343" #car
]
category_parent_name = "02958343"
cat_name = { #https://arxiv.org/pdf/1512.03012.pdf [page 8]
	"02691156" : "airplane",
        "02690373" :  "airliner",
        # "02828884",
	# "02933112",
	"03001627" : "chair",
	# "03211117",
	# "03636649",
	# "03691459",
	# "04090263",
	"04256520" : "sofa",
	# "04379243",
	# "04401088",
	# "04530566",
	"02958343" : "car"
}

class pcd_writer(DataFlow):
	def __init__(self, file_names):
		self.plylist = file_names
		# we apply a global shuffling here because later we'll only use local shuffling
		np.random.shuffle(self.plylist)
	def get_data(self):
		for ply_name in self.plylist:
			try:
				plydata = PlyData.read(ply_name)
				gc.collect()

			except ValueError:
				print '++++++ Oops! Discrading file length %s'%fname
				continue
			pcd = np.concatenate((np.expand_dims(plydata['vertex']['x'], 1), np.expand_dims(plydata['vertex']['z'], 1), np.expand_dims(plydata['vertex']['y'], 1)), 1)
			pcd = np.asarray(pcd, dtype='float32')
			# print np.mean(pcd, axis=0), np.mean(pcd, axis=0).shape, np.amax(pcd, axis=0), np.amin(pcd, axis=0)
			yield [pcd]
	def size(self):
		return len(self.plylist)


def get_ply_files(category_name, sample_num = 4096, splits = ['train', 'test', 'val']):
	model_ids = []
	file_names = []
	for split in splits:
		listFile = "./render_scripts/lists/PTNlist_v2/%s_%sids.txt"%(category_name, split)
		print listFile
		with open(listFile) as file:
			for line in file:
				model = line.strip()
				model_ids.append(model)
	model_ids.sort()
	modelN = len(model_ids)
	print '+++ Working on category %s with %d models in total...'%(category_name, modelN)

	path_read = "/home/rz1/Documents/Research/3dv2017_PBA_out/PCDs_v3/%s"%category_parent_name

	for i, model_id in enumerate(model_ids):
		ply_name = '%s/%s_%d.ply'%(path_read, model_id, sample_num)
		file_names.append(ply_name)
		# plydata = PlyData.read(ply_name)
		# pcd = np.concatenate((np.expand_dims(plydata['vertex']['x'], 1), np.expand_dims(plydata['vertex']['y'], 1), np.expand_dims(plydata['vertex']['z'], 1)), 1)
		# print pcd.shape
	return file_names


# category_name = "02691156"
splits_list = [['train', 'val'], ['test']]
for category_name in categories:
	for splits in splits_list:		
		if splits == ['train', 'test', 'val']:
			lmdb_name_append = 'all'
		else:
			if splits == ['train', 'val']:
				lmdb_name_append = 'train'
			else:
				if splits == ['test']:
					lmdb_name_append = 'test'
				else:
					print '++++++ Oops! Split cases not valid!', splits
					sys.exit(0)
		lmdb_write = "/home/rz1/Documents/Research/3dv2017_PBA/data/lmdb/%s_%d_%s_v3.lmdb"%(cat_name[category_name], sample_num, lmdb_name_append)

		command = 'rm -rf %s'%lmdb_write
		print command
		os.system(command)

		print "====== Writing to %s"%lmdb_write
		file_names = get_ply_files(category_name, sample_num = sample_num, splits = splits)
		ds0 = pcd_writer(file_names)
		file_names = file_names[:20]
		# ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
		dftools.dump_dataflow_to_lmdb(ds0, lmdb_write)
            