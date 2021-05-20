
import re 
import glob
import math
import cv2 
import numpy as np 
import torch
import torch.nn as nn
import imageio

from motioncompensation import align_frames
from models import DVDnet_temporal
from dvdnet import temporal_denoise 

# Parameters
noise_std = 0.5
# path2save = "results/denoised_P010.jpg"
path2save = "results/denoised_DAVIS_noisy_eval5.jpg"

device = "cuda"
# resume_path = "/home/abel/Desktop/repo_history/k_dvdnet/checkpoints/model_temp.pth"
resume_path = "/home/abel/Desktop/repo_history/k_dvdnet/checkpoints/2021-05-16_22-37-57/checkpoint-step-2000_2021-05-16_23-42-17.pth"

numframes = 5

# Functions
def crop(img):
    return img[:480,:854,]

# Load the test data
# dir2frames = "/home/abel/Desktop/repo_history/k_dvdnet/data/data_yuvP010_checkpoint-step-100000_2021-05-11_05-42-59/"
# listPath2frames = glob.glob(dir2frames+"*jpg")
# pobj = re.compile(r'.*_req[[]([0-9]*)[]]_batch.*_20210102_.*yuv_p010_100k.jpg')
# mobj = pobj.match(listPath2frames[20])

# def get_order(path):
#     match = pobj.match(path)
#     if not match:
#         return math.inf
#     return int(match.group(1))

# sorted_listPath = sorted(listPath2frames, key=get_order)

dir2frames = "/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS_noisy_eval5/"
listPath2frames = glob.glob(dir2frames+"*png")
sorted_listPath = sorted(listPath2frames)

# Get 5 frames for a quick test
frame_ref = cv2.imread(listPath2frames[2])
frame_ref = crop(frame_ref)
frame_ref = (cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB)).transpose(2,0,1)
C, H, W = frame_ref.shape
print("h={},w={}".format(H, W))
inframes = np.zeros((numframes, C, H, W))
inframes[2,:,:,:] = frame_ref
# Read 5 frames
print("Now loading the frames...")
for i in range(numframes):
    frame = cv2.imread(listPath2frames[i])
    frame = crop(frame)
    inframes[i,:,:,:] = ((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)/255.0).transpose(2,0,1))

# Align 5 frames
print("Now aligning the frames...")
inframes_wrpd = inframes.copy()
for i in range(numframes):
    if not i==2: 
        frame = inframes[i,:,:,:]
        inframes_wrpd[i,:,:,:] = (align_frames(frame.transpose((1,2,0)), frame_ref.transpose((1,2,0)), mc_alg='DeepFlow')/255.0).transpose(2,0,1)

# Convert the test data to tensors
inframes_wrpd = torch.from_numpy(inframes_wrpd).float()
print("inframes_wrpd.shape=", inframes_wrpd.shape)
inframes_wrpd = inframes_wrpd.contiguous().view((1, numframes*C, H, W)).to(device)
noise_map = torch.FloatTensor([noise_std]).expand((1, C, H, W)).to(device)

# Input to NN
print("Now loading the pretrained model...")
net_dvdT = DVDnet_temporal(num_input_frames=numframes).to(device).eval()
# net_dvdT = nn.DataParallel(net_dvdT, device_ids=[0]).cuda()

state_temp_dict = torch.load(resume_path)
net_dvdT.load_state_dict(state_temp_dict['net_dvdT'])

# Run the model
print("Now running the temporal_denoise model...")
print("torch.max(inframes_wrpd)=", torch.max(inframes_wrpd))
pred = temporal_denoise(model=net_dvdT, noisyframe=inframes_wrpd, sigma_noise=noise_map)

# Postprocess the output
inframes_wrpd = inframes_wrpd.view((numframes, C, H, W))
inframes_wrpd_np = inframes_wrpd.cpu().numpy().transpose((0,2,3,1))
pred_np = pred.cpu().detach().numpy().transpose((0,2,3,1))
imageio.imwrite(path2save, np.hstack((inframes_wrpd_np[2,:,:,:], pred_np[0,:,:,:])))

