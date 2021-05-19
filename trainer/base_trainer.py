import torch
from abc import abstractmethod
from utils import inf_loop
from torch.utils.tensorboard import SummaryWriter
from logger import TensorboardWriter
import os
import datetime

class BaseTrainer:
    def __init__(self, train_loader, config,
                 valid_loader=None, lr_scheduler=None, max_steps=None):
        self.config = config
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(config['device'])
        
        self.train_loader = inf_loop(train_loader)
        if max_steps is None:
            self.max_steps = 1000000
        else:
            self.max_steps = max_steps        
        
        self.start_step = 1
    
        self.log_dir = os.path.join(self.config['log_dir'],
                                        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if (config['resume'] is None) or (not config['resume']):
            self.checkpoint_dir = os.path.join(self.config['checkpoint_dir'], 
                                            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))            
        else:
            self._resume_checkpoint(self.config['resume_path'])
        
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):        
        it = iter(self.train_loader)
        for step in range(self.start_step, self.max_steps + 1):            
            result = self._train_step(next(it), step)
            if step % self.config['checkpoint_step'] == 0:
                self._save_checkpoint(step)

    def _train_step(self, input, step):
        raise NotImplementedError

    def _save_checkpoint(self, step, save_best=False):
        raise NotImplementedError
    
    def _resume_checkpoint(self, resume_path):
        raise NotImplementedError