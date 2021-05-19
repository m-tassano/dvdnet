#!/bin/sh
"""
Denoise all the sequences existent in a given folder using DVDnet.

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from models import DVDnet_spatial, DVDnet_temporal
from dvdnet import denoise_seq_dvdnet 
from utils.utils_DAVIS import batch_psnr, init_logger_test, variable_to_cv2_image, \
				remove_dataparallel_wrapper, open_sequence, close_logger

NUM_IN_FRAMES = 5 # temporal size of patch
MC_ALGO = 'DeepFlow' # motion estimation algorithm
OUTIMGEXT = '.png' # output images format

def save_out_seq(seqnoisy, seqclean, save_dir, sigmaval, suffix, save_noisy):
	"""Saves the denoised and noisy sequences under save_dir
	"""
	seq_len = seqnoisy.size()[0]
	for idx in range(seq_len):
		# Build Outname
		fext = OUTIMGEXT
		noisy_name = os.path.join(save_dir,\
						('n{}_{}').format(sigmaval, idx) + fext)
		if len(suffix) == 0:
			out_name = os.path.join(save_dir,\
					('n{}_DVDnet_{}').format(sigmaval, idx) + fext)
		else:
			out_name = os.path.join(save_dir,\
					('n{}_DVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

		# Save result
		if save_noisy:
			noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
			cv2.imwrite(noisy_name, noisyimg)

		outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
		cv2.imwrite(out_name, outimg)

def test_dvdnet(**args):
	"""Denoises all sequences present in a given folder. Sequences must be stored as numbered
	image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

	Inputs:
		args (dict) fields:
			"model_spatial_file": path to model of the pretrained spatial denoiser
			"model_temp_file": path to model of the pretrained temporal denoiser
			"test_path": path to sequence to denoise
			"suffix": suffix to add to output name
			"max_num_fr_per_seq": max number of frames to load per sequence
			"noise_sigma": noise level used on test set
			"dont_save_results: if True, don't save output images
			"no_gpu": if True, run model on CPU
			"save_path": where to save outputs as png
	"""
	start_time = time.time()

	# If save_path does not exist, create it
	if not os.path.exists(args['save_path']):
		os.makedirs(args['save_path'])
	logger = init_logger_test(args['save_path'])

	# Sets data type according to CPU or GPU modes
	if args['cuda']:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# Create models
	model_spa = DVDnet_spatial()
	model_temp = DVDnet_temporal(num_input_frames=NUM_IN_FRAMES)

	# Load saved weights
	state_spatial_dict = torch.load(args['model_spatial_file'])
	state_temp_dict = torch.load(args['model_temp_file'])
	if args['cuda']:
		device_ids = [0]
		model_spa = nn.DataParallel(model_spa, device_ids=device_ids).cuda()
		model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
	else:
		# CPU mode: remove the DataParallel wrapper
		state_spatial_dict = remove_dataparallel_wrapper(state_spatial_dict)
		state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
	model_spa.load_state_dict(state_spatial_dict)
	model_temp.load_state_dict(state_temp_dict)

	# Sets the model in evaluation mode (e.g. it removes BN)
	model_spa.eval()
	model_temp.eval()

	with torch.no_grad():
		# process data
		seq, _, _ = open_sequence(args['test_path'],\
									False,\
									expand_if_needed=False,\
									max_num_fr=args['max_num_fr_per_seq'])
		seq = torch.from_numpy(seq[:, np.newaxis, :, :, :]).to(device)

		seqload_time = time.time()

		# Add noise
		noise = torch.empty_like(seq).normal_(mean=0, std=args['noise_sigma']).to(device) # torch.Tensor.normal_() - in-place version of torch.normal()
		seqn = seq + noise
		noisestd = torch.FloatTensor([args['noise_sigma']]).to(device)

		denframes = denoise_seq_dvdnet(seq=seqn,\
										noise_std=noisestd,\
										temp_psz=NUM_IN_FRAMES,\
										model_temporal=model_temp,\
										model_spatial=model_spa,\
										mc_algo=MC_ALGO)
		den_time = time.time()

	# Compute PSNR and log it
	psnr = batch_psnr(denframes, seq.squeeze(), 1.)
	psnr_noisy = batch_psnr(seqn.squeeze(), seq.squeeze(), 1.)
	print("\tPSNR on {} : {}\n".format(os.path.split(args['test_path'])[-1], psnr))
	print("\tDenoising time: {:.2f}s".format(den_time - seqload_time))
	print("\tSequence loaded in : {:.2f}s".format(seqload_time - start_time))
	print("\tTotal time: {:.2f}s\n".format(den_time - start_time))
	logger.info("%s, %s, PSNR noisy %fdB, PSNR %f dB" % \
			 (args['test_path'], args['suffix'], psnr_noisy, psnr))

	# Save outputs
	if not args['dont_save_results']:
		# Save sequence
		save_out_seq(seqn, denframes, args['save_path'], int(args['noise_sigma']*255), \
					   args['suffix'], args['save_noisy'])

	# close logger
	close_logger(logger)

if __name__ == "__main__":
	import time 

	# Parse arguments
	parser = argparse.ArgumentParser(description="Denoise a sequence with DVDnet")
	parser.add_argument("--model_spatial_file", type=str,\
						default="checkpoints/model_spatial.pth", \
						help='path to model of the pretrained spatial denoiser')
	parser.add_argument("--model_temp_file", type=str,\
						default="checkpoints/model_temp.pth", \
						help='path to model of the pretrained temporal denoiser')
	# parser.add_argument("--test_path", type=str, default="./data/DAVIS/JPEGImages/480p/car-race", \
	# 					help='path to sequence to denoise')
	parser.add_argument("--test_path", type=str, default="/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS_noisy_eval5", \
						help='path to sequence to denoise')
	parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
	parser.add_argument("--max_num_fr_per_seq", type=int, default=100, \
						help='max number of frames to load per sequence')
	parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
	parser.add_argument("--save_noisy", action='store_true', default=True, help="save noisy images as well")
	parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
	parser.add_argument("--save_path", type=str, default='./results', \
						 help='where to save outputs as png')

	argspar = parser.parse_args()
	# Normalize noises to [0, 1]
	argspar.noise_sigma /= 255.

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing DVDnet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	start_time = time.time()
	test_dvdnet(**vars(argspar))
	elapsed_time = time.time() - start_time
	print("elapsed_time=", elapsed_time)