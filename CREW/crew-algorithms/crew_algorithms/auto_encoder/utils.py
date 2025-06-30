import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split

from crew_algorithms.auto_encoder import AutoEncoder
from crew_algorithms.utils.rl_utils import (
    convert_tensor_to_pil_image,
    unsqueeze_images_from_channel_dimension,
)
from torchvision.transforms import Resize, ToTensor, Compose
from crew_algorithms.auto_encoder.environment_dataset import EnvironmentDataset
import torch

from time import time


def make_dataloaders(cfg):
    """Makes a training, validation, and testing dataloader.

    Args:
        cfg: The configuration to use for making the dataloaders.
        dataset: The dataset for making the dataloaders.

    Returns:
        The train dataloader.
        The validation dataloader.
        The test dataloader.
    """
    transform = Compose([Resize((128, 128)), ToTensor()])
    train_set = EnvironmentDataset(cfg.data_root, transform, split="train", val_ratio=cfg.val_ratio)
    val_set = EnvironmentDataset(cfg.data_root, transform, split="val", val_ratio=cfg.val_ratio)

    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_dataloader, val_dataloader


def make_model(num_channels: int, embedding_dim: int):
    """Makes the AutoEncoder model.

    Args:
        num_channels: The number of channels the AutoEncoder should be
            created with.

    Returns:
        The AutoEncoder model.
    """
    net = AutoEncoder(num_channels, embedding_dim)
    return net


def make_loss():
    """Creates the loss used for the AutoEncoder.

    Returns:
        The loss to be used with the AutoEncoder.
    """
    return nn.MSELoss()


def make_optim(cfg, model: AutoEncoder):
    """Creates the optimizer used for the AutoEncoder.

    Args:
        cfg: The configuration settings.
        model: The model to optimize.

    Returns:
        The optimizer to be used with the AutoEncoder.
        The scheduler to be used with the AutoEncoder.
    """
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.scheduler_milestones, gamma=cfg.scheduler_gamma
    )
    return optimizer, scheduler

def train(cfg, train_loader, val_loader, model, loss_fn, optimizer, lr_scheduler, device):
    model.to(device)
    optimzer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    tic = time()
    for e in range(cfg.max_epochs):
        model.train()
        run_train, run_val = 0, 0
        for i, data in enumerate(train_loader):
            data = data.to(device)
            data_hat = model(data)
            loss = loss_fn(data_hat, data)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            run_train += loss.item()
            if (i+1) % cfg.log_freq == 0:
                print("ep %d| it %d/%d| train loss: %.7f| %dmin%ds" %(e, i, len(train_loader), run_train / cfg.log_freq, int((time()-tic)//60), int((time()-tic)%60)))
                run_train = 0

        for i, data in enumerate(val_loader):
            model.eval()
            data = data.to(device)
            with torch.inference_mode():
                data_hat = model(data)
            loss = loss_fn(data_hat, data)
            run_val += loss.item()
        print("EP %d| It %d/%d| val loss: %.7f" %(e, i, len(val_loader), run_val/(len(val_loader))))
        
        if (e+1) % cfg.save_freq == 0:
            torch.save(model.state_dict(), 'crew_algorithms/auto_encoder/weights/%d.pth'%e)

        lr_scheduler.step()
