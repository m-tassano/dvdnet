import torch
from trainer.trainer import Trainer
from data_loader.Dataset_DAVIS_aligned import Dataset_DAVIS_aligned
from data_loader.data_loaders import Dataloader_DAVIS


def main(config):
    train_dataset = Dataset_DAVIS_aligned(config['train_data_dir'])
    train_loader = Dataloader_DAVIS(dataset=train_dataset, batch_size=1, shuffle=False, validation_split=0.0, num_workers=1, training=True)


    valid_dataset = Dataset_DAVIS_aligned(config['valid_data_dir'])
    valid_loader = Dataloader_DAVIS(dataset=valid_dataset, batch_size=1, shuffle=False, validation_split=0.0, num_workers=1, training=False)

    trainer = Trainer(train_loader=train_loader, config=config, valid_loader=valid_loader)
    trainer.train()


if __name__ == "__main__":
    config = {
        'resume' : False,
        'resume_path' : 'checkpoints/checkpoint-step-20000_2021-05-11_13-13-42.pth',
        'train_data_dir' : "/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS/JPEGImages/480p",
        'valid_data_dir' : "/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS/JPEGImages/480p",
        'device': 'cuda',
        'checkpoint_dir' : 'checkpoints/',
        'log_dir' : 'logs/',
        'learning_rate' : 1e-4,
        'batch_size' : 1,
        'log_step' : 20,
        'checkpoint_step' : 2000,
        'valid_step' : 20
    }
    main(config)
    