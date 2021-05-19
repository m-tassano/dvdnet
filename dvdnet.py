"""
DVDnet denoising algorithm

@author: Matias Tassano <mtassano@parisdescartes.fr>
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils.utils_DAVIS import variable_to_cv2_image, normalize
from motioncompensation import align_frames

def temporal_denoise(model, noisyframe, sigma_noise):
	'''Encapsulates call to temporal model adding padding if necessary
	'''
	# Handle odd sizes
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%2
	expanded_w = sh_im[-1]%2
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

	# denoise
	out = torch.clamp(model(noisyframe, sigma_noise), 0., 1.)

	if expanded_h:
		out = out[:, :, :-1, :]
	if expanded_w:
		out = out[:, :, :, :-1]

	return out

def spatial_denoise(model, noisyframe, noise_map):
	'''Encapsulates call to spatial model adding padding if necessary
	'''
	# Handle odd sizes
	sh_im = noisyframe.size()
	expanded_h = sh_im[-2]%2
	expanded_w = sh_im[-1]%2
	padexp = (0, expanded_w, 0, expanded_h)
	noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
	noise_map = F.pad(input=noise_map, pad=padexp, mode='reflect')

	# denoise
	# out = torch.clamp(model(noisyframe, noise_map), 0., 1.)
	out = torch.clamp(noisyframe, 0., 1.)

	if expanded_h:
		out = out[:, :, :-1, :]
	if expanded_w:
		out = out[:, :, :, :-1]

	return out

def denoise_seq_dvdnet(seq, noise_std, temp_psz, model_temporal, model_spatial, mc_algo):
	r"""Denoises a sequence of frames with DVDnet.

	Args:
		seq: Tensor. [numframes, 1, C, H, W] array containing the noisy input frames
		noise_std: Tensor. Standard deviation of the added noise
		temp_psz: size of the temporal patch
		model_temp: instance of the PyTorch model of the temporal denoiser
		spatial_temp: instance of the PyTorch model of the spatial denoiser
		mc_algo: motion compensation algorithm to apply
	"""
	# init arrays to handle contiguous frames and related patches
	numframes, _, C, H, W = seq.shape
	ctrlfr_idx = int((temp_psz-1)//2)
	inframes = list()
	inframes_wrpd = np.empty((temp_psz, H, W, C))
	denframes = torch.empty((numframes, C, H, W)).to(seq.device)

	# build noise map from noise std---assuming Gaussian noise
	noise_map = noise_std.expand((1, C, H, W))

	for fridx in range(numframes):
		# load input frames
		# denoise each frame with spatial denoiser when appending
		if not inframes:
		# if list not yet created, fill it with temp_patchsz frames
			for idx in range(temp_psz):
				relidx = max(0, idx-ctrlfr_idx)
				inframes.append(spatial_denoise(model_spatial, seq[relidx], noise_map))
		else:
			del inframes[0]
			relidx = min(numframes-1, fridx+ctrlfr_idx)
			inframes.append(spatial_denoise(model_spatial, seq[relidx], noise_map))

		# save converted central frame
		# OpenCV images are HxWxC uint8 images
		inframes_wrpd[ctrlfr_idx] = variable_to_cv2_image(inframes[ctrlfr_idx], conv_rgb_to_bgr=False)

		# register frames w.r.t central frame
		# need to convert them to OpenCV images first
		for idx in range(temp_psz):
			if not idx == ctrlfr_idx:
				img_to_warp = variable_to_cv2_image(inframes[idx], conv_rgb_to_bgr=False)
				inframes_wrpd[idx] = align_frames(img_to_warp, \
												inframes_wrpd[ctrlfr_idx], \
												mc_alg=mc_algo)
		# denoise with temporal model
		# temp_pszxHxWxC to temp_pszxCxHxW
		inframes_t = normalize(inframes_wrpd.transpose(0, 3, 1, 2))
		inframes_t = torch.from_numpy(inframes_t).contiguous().view((1, temp_psz*C, H, W)).to(seq.device)

		# append result to output list
		denframes[fridx] = temporal_denoise(model_temporal, inframes_t, noise_map)

	# free memory up
	del inframes
	del inframes_wrpd
	del inframes_t
	torch.cuda.empty_cache()

	# convert to appropiate type and return
	return denframes
