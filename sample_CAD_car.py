import scipy.io
import numpy as np
import os,sys
import time
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../models/chen-hsuan'))
import util as util

categories = [
	# "02691156", #airplane
	# "02828884",
	# "02933112",
	# "03001627", #chair
	# "03211117",
	# "03636649",
	# "03691459",
	# "04090263",
	# "04256520", # sofa
	# "04379243",
	# "04401088",
	# "04530566",
	"02958343" #car
]
sample_binary = "/home/rz1/Documents/Research/3dv2017_PBA/utils/pcl_mod/build2/bin/pcl_mesh_sampling"
# sample_binary = "/home/rz1/Documents/Research/3dv2017_PBA/data/pcl/build/bin/pcl_uniform_sampling"
sample_num = 24576
pool_num = 28

for category_name in categories:
	path_write = "/home/rz1/Documents/Research/3dv2017_PBA_out/PCDs_v3/%s"%category_name
	if not os.path.exists(path_write):
		os.mkdir(path_write)

	path_read = "/dataset/ShapeNetCore.v1/%s"%category_name

	model_ids = []
	for split in ['train', 'test', 'val']:
		listFile = "/home/rz1/Documents/Research/3dv2017_PBA/data/render_scripts/lists/PTNlist_v2/%s_%sids.txt"%(category_name, split)
				# listFile = "/home/chenhsuan/3dgen/data/PTNlist/%s_%sids.txt"%(category_name, split)
		with open(listFile) as file:
			for line in file:
				# model = line.strip().split("/")[1] #v1 list
				model = line.strip() #v2 list
				model_ids.append(model)
	model_ids.sort()
	# model_ids = os.listdir(path_read)
	modelN = len(model_ids)
	print '+++ Working on category %s with %d models in total...'%(category_name, modelN)

	commands = []
	for i, model_id in enumerate(model_ids):
		# print model_id
		input_name = '%s/%s/model.obj'%(path_read, model_id)
		output_name = '%s/%s_%d'%(path_write, model_id, sample_num)
		command_gen = "%s %s %s.ply %s.nor %s.fid -n_samples %d -leaf_size 0.01 -visu 0"\
		%(sample_binary, input_name, output_name, output_name, output_name, sample_num)
		# os.system(command_gen)
		# print command_gen
		# print '-- Written %d/%d: %s'%(i, modelN, output_name)
		commands.append(command_gen)

	print(util.toMagenta('=== Sampling %d CADs on %d workers, it takes a long time...'%(len(commands), pool_num)))
	pool = Pool(pool_num)
	for idx, return_code in enumerate(pool.imap(partial(call, shell=True), commands)):
		print(util.toBlue('-- Written %d/%d: %s'%(idx, modelN, output_name)))
		if return_code != 0:
			print(util.toYellow('Rendering command %d of %d (\"%s\") failed' % (idx, len(commands), commands[idx])))



                        