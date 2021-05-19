import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
print("currentdir=", currentdir)
parentdir = os.path.dirname(currentdir)
print("parentdir=", parentdir)
sys.path.append(parentdir)

import torch
import glob
import numpy as np 
from utils.utils_DAVIS import open_image

class Dataset_DAVIS(torch.utils.data.Dataset):
    """
    DAVIS dataset
    temp_psz: the number of frames as the input for the NN, default=5
    """

    def __init__(self, path2videos, temp_psz=5, transform=None, noise_sigma=0.1, device='cpu'):
        """
        Args:
            path2videos (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.device = device
        self.path2videos = path2videos
        self.temp_psz = temp_psz
        self.ctrlfr_idx = int((temp_psz-1)//2)
        self.transform = transform
        self.noise_sigma = noise_sigma
        self.list_videoFolders = sorted(glob.glob(self.path2videos+"/*"))
        self.num_videos = len(self.list_videoFolders)
        self.arr_videoFrameNum = np.zeros(self.num_videos, dtype='i4') # 4-byte int
        # print("self.ctrlfr_idx=", self.ctrlfr_idx)
        idxVideo = 0
        for videoFolder in self.list_videoFolders:
            list_img = glob.glob(videoFolder+"/*jpg")
            self.arr_videoFrameNum[idxVideo] = len(list_img)
            idxVideo += 1
        # print(self.arr_videoFrameNum)

    def __len__(self):
        return np.sum(self.arr_videoFrameNum)

    def __getitem__(self, idx):
        arr_accumFrameNum = np.add.accumulate(self.arr_videoFrameNum)
        # print("arr_accumFrameNum=", arr_accumFrameNum)
        idxVideo = np.searchsorted(arr_accumFrameNum, idx)
        # print("idxVideo=", idxVideo)
        if idxVideo==0:
            idxFrame = idx
        else:
            idxFrame = idx - arr_accumFrameNum[idxVideo-1]-1 # starts from 0
        videoFolder = self.list_videoFolders[idxVideo]
        # print("videoFolder=", videoFolder)
        list_img = sorted(glob.glob(videoFolder+"/*jpg"))
        inframes = list()
        for i in range(self.temp_psz):
            if idxFrame-self.ctrlfr_idx+i<0:
                relidx = 0
            elif idxFrame-self.ctrlfr_idx+i>len(list_img)-1:
                relidx = len(list_img)-1
            else:
                relidx = idxFrame-self.ctrlfr_idx+i
            img, expanded_h, expanded_w = open_image(list_img[relidx],\
                                                gray_mode=False, expand_axis0=True)
            inframes.append(img)
        # list to numpy array
        # inframes = np.stack(inframes, axis=0)
        inframes = np.array(inframes)
        # print("numpy inframes.shape=", inframes.shape)
        # numpy to torch tensor
        inframes = torch.from_numpy(inframes).to(self.device)
        # target
        target = inframes[self.ctrlfr_idx]
        # Add noise
        noise = torch.empty_like(inframes).normal_(mean=0, std=self.noise_sigma).to(self.device)
        inframes = inframes + noise
        noise_std = torch.FloatTensor([self.noise_sigma]).to(self.device)
        # print("tensor inframes.shape=", inframes.shape)
        return inframes, target, noise_std



if __name__=="__main__":
    import imageio 

    path2videos = "/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS/JPEGImages/480p"
    list_videoFolders = glob.glob(path2videos+"/*")
    print(len(list_videoFolders))
    # print(list_videoFolders)

    dataset = Dataset_DAVIS(path2videos)
    # print("dataset.shape=", dataset.shape)
    print(len(dataset))

    inframes, target, noise_std = dataset[5]
    print(len(inframes))
    print("noise_std=", noise_std)

    # export target
    target_np = target.squeeze().numpy().transpose((1,2,0))
    imageio.imwrite("data_loader/test_export_target.jpg", target_np)

    # export inframes
    for i in range(len(inframes)):
        frame_t = inframes[i]
        print("frame_t.shape=", frame_t.shape) # (1, 3, 480, 854)
        frame_np = frame_t.squeeze().numpy().transpose((1,2,0))
        print("frame_np.shape=", frame_np.shape) # (480, 854, 3)
        imageio.imwrite("data_loader/test_export_inframes_index{}.jpg".format(i), frame_np)

