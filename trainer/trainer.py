import torch
import torch.nn as nn

from abc import abstractmethod
from utils import inf_loop
from torch.utils.tensorboard import SummaryWriter
# from logger import TensorboardWriter
import os
import datetime
from models import DVDnet_temporal
from dvdnet import temporal_denoise 
import imageio
import numpy as np

class Trainer:
    def __init__(self, train_loader, config,
                 valid_loader=None, lr_scheduler=None, max_steps=None):
        self.config = config
        self.device = torch.device(config['device'])
        self.train_loader = inf_loop(train_loader)
        self.valid_loader = valid_loader
        if max_steps is None:
            self.max_steps = 1000000
        else:
            self.max_steps = max_steps        
        
        self.start_step = 1

        # Prepare network
        self.net_dvdT = DVDnet_temporal(num_input_frames=5).to(self.device).train()
        # print("self.net_dvdT=", self.net_dvdT)
        # Define optimizer
        self.optim = torch.optim.Adam(self.net_dvdT.parameters(), lr=self.config['learning_rate'], betas=(0.5, 0.99))
        # print("self.optim=", self.optim)
        # Define loss
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = nn.MSELoss()

        # self.log_dir = os.path.join(self.config['log_dir'],
        #                                 datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if (config['resume'] is None) or (not config['resume']):
            self.checkpoint_dir = os.path.join(self.config['checkpoint_dir'], 
                                            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))            
        else:
            self._resume_checkpoint(self.config['resume_path'])
        
        # self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):        
        it_train = iter(self.train_loader)
        it_valid = iter(self.valid_loader)
        for step in range(self.start_step, self.max_steps + 1):            
            self._train_step(next(it_train), step)
            if step % self.config['checkpoint_step'] == 0:
                self._save_checkpoint(step)

    def _train_step(self, input, step):
        inframes, target, noise_std = input
        inframes = inframes.squeeze(dim=0).squeeze(dim=1)
        target = target.squeeze(dim=0).squeeze(dim=1)
        numframes, C, H, W = inframes.shape
        # build noise map from noise std---assuming Gaussian noise
        noise_map = noise_std.expand((1, C, H, W))
        # reshape inframes
        inframes = inframes.contiguous().view((1, numframes*C, H, W))
        # transfer data
        inframes = inframes.to(self.device)
        target = target.to(self.device)
        noise_map = noise_map.to(self.device)
        # run the model
        pred = temporal_denoise(model=self.net_dvdT, noisyframe=inframes, sigma_noise=noise_map)
        # calculating loss
        loss_l1 = self.criterionL1(pred, target)
        # back-prop
        self.optim.zero_grad()
        loss_l1.backward()
        self.optim.step()
        # print info
        if step % self.config["log_step"]==0:
            print("step-{} | loss_l1: {:.5f}".format(step, loss_l1))

    def _save_checkpoint(self, step):
        state = {
            'step' : step,
            'net_dvdT': self.net_dvdT.state_dict(),
            'checkpoint_dir' : self.checkpoint_dir
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-step-{}_{}.pth'.format(step, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(state, filename)
    
    def _resume_checkpoint(self, resume_path):
        resume_path = self.config['resume_path']
        checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        self.start_step = checkpoint['step'] + 1
        self.net_dvdT.load_state_dict(checkpoint['net_dvdT'])
        self.checkpoint_dir = self.config['checkpoint_dir']
        # self.checkpoint_dir = os.path.join(self.checkpoint_dir, 'checkpoint-step-{}_{}.pth'.format(step, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        print("Completed loading the checkpoint")

    def test(self, path2save):
        it_valid = iter(self.valid_loader)
        inframes_wrpd, target, noise_std = next(it_valid)
        with torch.no_grad():
            inframes_wrpd = inframes_wrpd.squeeze(dim=0).squeeze(dim=1)
            target = target.squeeze(dim=0).squeeze(dim=1)
            numframes, C, H, W = inframes_wrpd.shape
            # build noise map from noise std---assuming Gaussian noise
            noise_map = noise_std.expand((1, C, H, W))
            # reshape inframes
            inframes_wrpd = inframes_wrpd.contiguous().view((1, numframes*C, H, W))
            # transfer data
            inframes_wrpd = inframes_wrpd.to(self.device)
            target = target.to(self.device)
            noise_map = noise_map.to(self.device)
            # run the model
            pred = temporal_denoise(model=self.net_dvdT, noisyframe=inframes_wrpd, sigma_noise=noise_map)
        # convert tensor to numpy
        inframes_wrpd = inframes_wrpd.view((numframes, C, H, W))
        inframes_wrpd_np = inframes_wrpd.cpu().numpy().transpose((0,2,3,1))
        target_np = target.cpu().numpy().transpose((0,2,3,1))
        pred_np = pred.cpu().numpy().transpose((0,2,3,1))
        imageio.imwrite(path2save, np.hstack((inframes_wrpd_np[2,:,:,:], target_np[0,:,:,:], pred_np[0,:,:,:])))
