"""
Definition of the DVDnet model

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import torch
import torch.nn as nn

class DVDnet_spatial(nn.Module):
	""" Definition of the spatial denoiser of DVDnet.
	Inputs of forward():
		x: array of input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, C, H, W], C (noise map for each channel)
	"""

	def __init__(self):
		super(DVDnet_spatial, self).__init__()
		self.down_kernel_size = (2, 2)
		self.down_stride = 2
		self.kernel_size = 3
		self.padding = 1
		# RGB image
		self.num_input_channels = 6
		self.middle_features = 96
		self.num_conv_layers = 12
		self.down_input_channels = 12
		self.downsampled_channels = 15
		self.output_features = 12

		self.downscale = nn.Unfold(kernel_size=self.down_kernel_size, stride=self.down_stride)

		layers = []
		layers.append(nn.Conv2d(in_channels=self.downsampled_channels,\
								out_channels=self.middle_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(self.num_conv_layers-2):
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=self.middle_features,\
								out_channels=self.output_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))

		self.conv_relu_bn = nn.Sequential(*layers)
		self.pixelshuffle = nn.PixelShuffle(2)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map):

		N, _, H, W = x.size() # compute size of input

		# Downscale input using nn.Unfold
		x1 = self.downscale(x)
		x1 = x1.reshape(N, self.down_input_channels, H//2, W//2)

		# Concat downscaled input with downsampled noise map
		x1 = torch.cat((noise_map[:, :, ::2, ::2], x1), 1)

		# Conv + ReLU + BN
		x1 = self.conv_relu_bn(x1)

		# Upscale back to original resolution
		x1 = self.pixelshuffle(x1)

		# Residual learning
		x = x - x1
		return x

class DVDnet_temporal(nn.Module):
	""" Definition of the temporal denoiser of DVDnet.
	Inputs of constructor:
		num_input_frames: int. number of frames to denoise
	Inputs of forward():
		x: array of input frames of dim [num_input_frames, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [1, C, H, W], C (noise map for each channel)
	"""
	def __init__(self, num_input_frames):
		super(DVDnet_temporal, self).__init__()
		self.num_input_frames = num_input_frames
		self.num_input_channels = int((num_input_frames+1)*3) # num_input_frames RGB frames + noisemap
		self.num_feature_maps = 96
		self.num_conv_layers = 4
		self.output_features = 12
		self.down_kernel_size = 5
		self.down_stride = 2
		self.down_padding = 2
		self.conv1x1_kernel_size = 1
		self.conv1x1_stride = 1
		self.conv1x1_padding = 0
		self.kernel_size = 3
		self.stride = 1
		self.padding = 1

		self.down_conv = nn.Sequential(nn.Conv2d(in_channels=self.num_input_channels,\
												out_channels=self.num_feature_maps,\
												kernel_size=self.down_kernel_size,\
												padding=self.down_padding,\
												stride=self.down_stride,\
												bias=False),\
										nn.BatchNorm2d(self.num_feature_maps),\
										nn.ReLU(inplace=True))
		self.conv1x1 = nn.Conv2d(in_channels=self.num_feature_maps,\
											out_channels=self.num_feature_maps,\
											kernel_size=self.conv1x1_kernel_size,\
											padding=self.conv1x1_padding,\
											stride=self.conv1x1_stride,\
											bias=False)
		layers = []
		for _ in range(self.num_conv_layers):
			layers.append(nn.Conv2d(in_channels=self.num_feature_maps,\
									out_channels=self.num_feature_maps,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			layers.append(nn.BatchNorm2d(self.num_feature_maps))
			layers.append(nn.ReLU(inplace=True))

		self.block_conv = nn.Sequential(*layers)
		self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.num_feature_maps,\
											out_channels=self.num_feature_maps,\
											kernel_size=self.kernel_size,\
											padding=self.padding,\
											stride=self.stride,\
											bias=False),\
										nn.BatchNorm2d(self.num_feature_maps),\
										nn.ReLU(inplace=True),\
										nn.Conv2d(in_channels=self.num_feature_maps,\
											out_channels=self.output_features,\
											kernel_size=self.kernel_size,\
											padding=self.padding,\
											stride=self.stride,\
											bias=False))
		self.pixelshuffle = nn.PixelShuffle(2)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map):
		x1 = torch.cat((noise_map, x), 1)
		x1 = self.down_conv(x1)
		x2 = self.conv1x1(x1)
		x1 = self.block_conv(x1)
		x1 = self.out_conv(x1+x2)
		x1 = self.pixelshuffle(x1)
		return x1
