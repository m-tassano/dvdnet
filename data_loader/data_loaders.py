# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# print("currentdir=", currentdir)
# parentdir = os.path.dirname(currentdir)
# print("parentdir=", parentdir)
# sys.path.append(parentdir)

from torchvision import datasets, transforms
# from data_loader.base_data_loader import BaseDataLoader
from base import BaseDataLoader

class Dataloader_DAVIS(BaseDataLoader):
    """
    data loading
    """
    def __init__(self, dataset, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.dataset = dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__=="__main__":
    from data_loader.Dataset_DAVIS_aligned import Dataset_DAVIS_aligned

    path2videos = "/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS/JPEGImages/480p"
    dataset = Dataset_DAVIS_aligned(path2videos)
    dataloader = Dataloader_DAVIS(dataset=dataset, batch_size=1, shuffle=False, validation_split=0.0, num_workers=1, training=True)
    for batch_idx, (inframes, target, noisestd) in enumerate(dataloader):
        print("batch_idx=", batch_idx)
        print("target.shape=", target.shape) # torch.Size([1, 1, 1, 3, 480, 854])
        print("inframes.shape=", inframes.shape) # torch.Size([1, 5, 1, 1, 3, 480, 854])
