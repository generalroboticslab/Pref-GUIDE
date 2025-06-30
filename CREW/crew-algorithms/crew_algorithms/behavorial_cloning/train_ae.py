import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from crew_algorithms.auto_encoder.model import AutoEncoder
from torchvision.utils import save_image
from torchvision.transforms import ToTensor

from pytorch_msssim import ssim, ms_ssim

totens = ToTensor()

class FindTreasureDataset(Dataset):
    def __init__(self, path, transform=None, split='train'):
        torch._assert(split in ['train', 'val'], 'split must be one of train, val')

        self.image_folder = path
        self.image_files = [f for f in os.listdir(self.image_folder)]
        self.len_all = len(self.image_files)

        self.data = self.image_files[:int(self.len_all * 0.9)] if split == 'train' else self.image_files[int(self.len_all * 0.9):]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        img_path = self.image_folder + img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

# Define transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((113, 113)),
    # transforms.RandomAffine(degrees=(0, 0), translate=(0.15, 0.15)),
    transforms.Resize((128, 128))
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((113, 113)),
    transforms.Resize((128, 128))
    ])
    

# Create an instance of the dataset
train_data = FindTreasureDataset('../Data/obs/', transform=transform, split='train')
val_data = FindTreasureDataset('../Data/obs/', transform=test_transform, split='val')
len_data = len(train_data)

model = AutoEncoder(channels=3, embedding_dim=384)

def train(bs, lr, wd, eps):
    train_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=bs, shuffle=False)
    # model.load_state_dict(torch.load('../Data/ae_models/198.pth'))
    model.to('cuda')
    optimzer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # optimzer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()

    for e in range(0, eps):
        model.train()
        run_train, run_val = 0, 0
        for i, data in enumerate(train_loader):
        
            data = data.to('cuda')
            data_hat = model(data)
            loss = loss_fn(data_hat, data)
            # loss = 1 - ssim(data_hat, data)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            run_train += loss.item()

            if (i+1) % 100 == 0:
                print("ep %d| it %d/%d| train loss: %.7f" %(e, i, len(train_loader), run_train / 100))
                run_train = 0

        for i, data in enumerate(val_loader):
            model.eval()
            data = data.to('cuda')
            with torch.inference_mode():
                data_hat = model(data)
            loss = loss_fn(data_hat, data)
            # loss = 1 - ssim(data_hat, data)
            run_val += loss.item()
            # if (i+1) % 10 == 0:
        print("ep %d| it %d/%d| val loss: %.7f" %(e, i, len(val_loader), run_val/(len(val_loader))))
        # run_val = 0
        
        torch.save(model.state_dict(), '../Data/ae_models/%d.pth'%e)


def test():
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False)
    model.eval()
    model.to('cuda')
    # model.load_state_dict(torch.load('crew_algorithms/auto_encoder/ae.pth'))
    model.load_state_dict(torch.load('../Data/ae_models/143.pth'))
    model.eval()



    for i, d in enumerate([('../Data/ae_check/' + f) for f in os.listdir('../Data/ae_check/')]):
        if 'hat' in d:
            continue
        d_name = d
        d = Image.open(d).convert('RGB')
        d = test_transform(d).unsqueeze(0)
        print(d.max())

        d = d.to('cuda')
        data_hat = model(d)
        save_image(d, d_name[:-4] + '_org.png')
        save_image(data_hat, d_name[:-4] + '_hat.png')

    
    # for i, d in enumerate(val_loader):
    #     d = d.to('cuda')
    #     data_hat = model(d)
    #     save_image(data_hat, '../Data/ae_models/data_hat%d.png'%i)
    #     save_image(d, '../Data/ae_models/data_%d.png'%(i))
    #     if i> 20: 
    #         break
    

if __name__ == '__main__':
    # train(64, 1e-4, 1e-5, 300)
    test()

