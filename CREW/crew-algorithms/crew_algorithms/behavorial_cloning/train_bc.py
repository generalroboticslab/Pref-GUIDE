import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from crew_algorithms.multimodal_feedback.policy import ActorNet
from crew_algorithms.auto_encoder.model import Encoder
import random

import math

totens = transforms.ToTensor()

def action_mapping(pos_1, pos_2, rot_1, rot_2):
    pos_diff = pos_2 - pos_1
    rot_diff = rot_2 - rot_1

    pos_vec = torch.tensor([pos_diff[..., 0], pos_diff[..., 2]])
    rot_vec = torch.tensor([math.cos(2*math.pi*((90-rot_1)/360)), math.sin(2*math.pi*((90-rot_1)/360))])

    x = torch.dot(pos_vec, rot_vec) / pos_vec.norm()
    actions = []
    # if x > 0.5:
    #     print(x)
    if x > 0.7:
        actions.append(0)
    elif x < -0.7:
        actions.append(1)
    if rot_diff > 0.1:
        actions.append(3)
    elif rot_diff < -0.1:
        actions.append(2)

    if len(actions) == 0:
        actions.append(0)
    return random.choice(actions)



class FindTreasurePolicyDataset(Dataset):
    def __init__(self, image_path, action_path, transform=None):
        self.image_folder = image_path
        self.image_files = [(self.image_folder + '/' + f) for f in os.listdir(self.image_folder)]#+ [('../Data/obs_low/' + f) for f in os.listdir('../Data/obs_low')]

        self.action_folder = action_path
        self.action_files = [(self.action_folder + '/' + f) for f in os.listdir(self.action_folder)]# + [('../Data/act_low/' + f) for f in os.listdir('../Data/act_low')]

        self.image_files.sort()
        self.action_files.sort()
        self.i = []
        self.a = []

        # balance classes
        # for i in range(len(self.action_files)):
        #     if torch.load(self.action_files[i]) == 0:
        #         if random.choice(range(4)) != 0:
        #             continue
        #     self.i.append(self.image_files[i])
        #     self.a.append(self.action_files[i])

        self.i = self.image_files#[::5]
        self.a = self.action_files#[::5]
        
        self.transform = transform

    def __len__(self):
        return len(self.i) - 1
    

    def __getitem__(self, idx):

        obs_path = self.i[idx]
        obs = Image.open(obs_path).convert('RGB')

        act_path = self.a[idx]
        action = torch.load(act_path)

        assert obs_path.split('/')[-1].split('.')[0][1:] == act_path.split('/')[-1].split('.')[0][1:],'Obs and action files do not match: %s, %s' %(obs_path, act_path)

        if self.transform:
            obs = self.transform(obs)

        # rand_idx = random.choice(range(4))
        # if rand_idx == 0:
        #     obs = torch.flip(obs, [1])
        #     if action == 2:
        #         action = 3
        #     elif action == 3:
        #         action = 2
        # elif rand_idx ==1:
        #     obs = torch.flip(obs, [2])
        #     if action == 2:
        #         action = 3
        #     elif action == 3:
        #         action = 2    

        # print(vec_path, act_path)

        # print(vec_path, act_path)


        # pos_name = self.positions_files[idx]
        # pos_path = self.positions_folder + str(idx) + '.pt'
        # position = torch.load(pos_path)
        # pos, rot = position[..., :3], position[..., 4]

        # # next_pos_name = self.positions_files[idx+1]
        # next_pos_path = self.positions_folder + str(idx+1) + '.pt'
        # next_position = torch.load(next_pos_path)
        # next_pos, next_rot = next_position[..., :3], next_position[..., 4]


        # action = action_mapping(pos, next_pos, rot, next_rot)
        
        # pos_diff = next_pos - pos
        # print(rot, pos_diff)
        # rot_diff = next_rot - rot
        # dx, dz = pos_diff[..., 0], pos_diff[..., 2]
        # print(rot_diff)
        # print(pos_diff, rot)

        # if rot_diff>0.0001:
        #     action = torch.tensor([3])
        # elif rot_diff<-0.0001:
        #     action = torch.tensor([2])
        # elif math.cos(2*math.pi*((90-rot)/360)) * dx + math.sin(2*math.pi*((90-rot)/360)) * dz >0:
        #     action = torch.tensor([0])
        # else:
        #     action = torch.tensor([1])

        # if pos_diff.norm() > 0.5:
        #     self.max = pos_diff.norm()
        #     self.restart_idx.append(idx)

        return obs, action#, pos_diff, rot

transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Grayscale(),
    # transforms.RandomRotation(180),
    # transforms.Resize((84, 84)),
    # transforms.Normalize((0.5,), (0.5,))
])

# Create an instance of the dataset
all_data = FindTreasurePolicyDataset('../Data/BC_data/obs', 
                                     '../Data/BC_data/act', transforms)
len_data = len(all_data)
print(len_data)

train_data = torch.utils.data.Subset(all_data, range(0, int(len_data * 0.90)))
val_data = torch.utils.data.Subset(all_data, range(int(len_data * 0.90), int(len_data * 1)))
# train_data, val_data = torch.utils.data.random_split(all_data, [int(len_data * 0.9), len_data - int(len_data * 0.9)])


encoder = Encoder(3, 64)
model = ActorNet(
        encoder = encoder,
        n_agent_inputs=64,
        num_cells=[256, 256],
        n_agent_outputs=4,
        activation_class=nn.ReLU,
    )

from torchvision.utils import save_image

def entropy(logits):
    p = torch.softmax(logits, dim=1)
    return -torch.sum(p * torch.log(p + 1e-8), dim=1)

def train(bs, lr, wd, eps):

    train_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=bs, shuffle=False)
    model.train()
    optimzer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0, 3.0, 3.0]).to('cuda'))
    loss_fn = torch.nn.CrossEntropyLoss()
    model.to('cuda')

    # model.load_state_dict(torch.load('../Data/actor_model3/ep1.pth'))
    # import pdb; pdb.set_trace()

    get_stats(val_loader)
    # v4 stats: 74.3%, 3.22%, 11.46%, 10.92%
    # v5 stats: 77.72%, 2.50%, 9.80%, 9.98% (train)
    # v5 stats: 76.77%, 2.68%, 9.34%, 11.21% (val)


    best_acc = 0
    best_loss = 99999999

    from time import time
    tic = time()

    for e in range(0, eps):
        run_train, train_total, train_correct, run_val, val_total, val_correct = [0] * 6
        train_per_action_correct = [0] * 4
        train_per_action_total = [0] * 4
        val_per_action_correct = [0] * 4
        val_per_action_total = [0] * 4
        for i, data in enumerate(train_loader):
            # v, a, p, r = data
            # print(p, r)
            model.train()
            v, a = data
            v = v.to('cuda').unsqueeze(1)
            a = a.to('cuda').type(torch.int64)

            # import pdb; pdb.set_trace()
            a_pred = model(v).squeeze(1).squeeze(1)
            loss = loss_fn(a_pred, a)

            entropy_loss = entropy(a_pred).mean()
            loss = loss - 0.3 * entropy_loss
            # print(loss.item(), entropy_loss.item())
            #

            optimzer.zero_grad()
            loss.backward()
            optimzer.step()

            run_train += loss.item()
            train_total += a.size(0)
            train_correct += (torch.argmax(a_pred, dim=1) == a).sum().item()

            train_per_action_correct[0] += ((torch.argmax(a_pred, dim=1) == a) * (a == 0)).sum().item()
            train_per_action_correct[1] += ((torch.argmax(a_pred, dim=1) == a) * (a == 1)).sum().item()
            train_per_action_correct[2] += ((torch.argmax(a_pred, dim=1) == a) * (a == 2)).sum().item()
            train_per_action_correct[3] += ((torch.argmax(a_pred, dim=1) == a) * (a == 3)).sum().item()

            train_per_action_total[0] += (a == 0).sum().item()
            train_per_action_total[1] += (a == 1).sum().item()
            train_per_action_total[2] += (a == 2).sum().item()
            train_per_action_total[3] += (a == 3).sum().item()


            # print(train_correct, train_total)

            if (i+1) % (len(train_loader)//2 - 1) == 0:
                print("ep %d| it %d/%d| train loss: %.5f| train_acc: %.2f%%| train_per_action_acc: [%.1f%%, %.1f%%, %.1f%%, %.1f%%]| %dm%ds" %(e, i, len(train_loader), run_train / 500, 100 * train_correct / train_total, 100 * train_per_action_correct[0] / train_per_action_total[0], 100 * train_per_action_correct[1] / train_per_action_total[1], 100 * train_per_action_correct[2] / train_per_action_total[2], 100 * train_per_action_correct[3] / train_per_action_total[3], (time()- tic) // 60, (time() - tic) % 60))
                run_train, train_total, train_correct = 0, 0, 0
                per_action_correct = [0] * 4
                per_action_total = [0] * 4

                for j, data in enumerate(val_loader):
                    model.eval()
                    with torch.inference_mode():
                        v, a = data
                        v = v.to('cuda').unsqueeze(1)
                        a = a.to('cuda').type(torch.int64)
                        a_pred = model(v).squeeze(1).squeeze(1)
                        loss = loss_fn(a_pred, a)
                        val_total += a.size(0)
                        val_correct += (torch.argmax(a_pred, dim=1) == a).sum().item()
                        val_per_action_correct[0] += ((torch.argmax(a_pred, dim=1) == a) * (a == 0)).sum().item()
                        val_per_action_correct[1] += ((torch.argmax(a_pred, dim=1) == a) * (a == 1)).sum().item()
                        val_per_action_correct[2] += ((torch.argmax(a_pred, dim=1) == a) * (a == 2)).sum().item()
                        val_per_action_correct[3] += ((torch.argmax(a_pred, dim=1) == a) * (a == 3)).sum().item()

                        val_per_action_total[0] += (a == 0).sum().item()
                        val_per_action_total[1] += (a == 1).sum().item()
                        val_per_action_total[2] += (a == 2).sum().item()
                        val_per_action_total[3] += (a == 3).sum().item()
                    run_val += loss.item()
                    # if (i+1) % 100 == 0:
                if True:#100 * val_correct / val_total > best_acc:
                    best_loss = run_val
                    #best_acc = 100 * val_correct / val_total
                    

                    print("ep %d| it %d/%d| val loss: %.5f| val_acc: %.2f%%| val_per_action_acc: [%.1f%%, %.1f%%, %.1f%%, %.1f%%]" %(e, j, len(val_loader), run_val / len(val_loader), 100 * val_correct / val_total, 100 * val_per_action_correct[0] / val_per_action_total[0], 100 * val_per_action_correct[1] / val_per_action_total[1], 100 * val_per_action_correct[2] / val_per_action_total[2], 100 * val_per_action_correct[3] / val_per_action_total[3]))
                    torch.save(model.state_dict(), '../Data/BC_data/actor_weights/ep%d_it%d.pth'%(e, i))
                    # if val_per_action_correct[2]/val_per_action_total[2] + val_per_action_correct[3]/val_per_action_total[3] > best_acc:
                    #     torch.save(model.state_dict(), '../Data/actor_model3/best.pth')
                    #     print('Saved best model')
                    #     best_acc = val_per_action_correct[2]/val_per_action_total[2] + val_per_action_correct[3]/val_per_action_total[3]
                run_val, val_total, val_correct = 0, 0, 0
                val_per_action_correct = [0] * 4
                val_per_action_total = [0] * 4

        # torch.save(model.state_dict(), '../Data/actor_model3/ep%d.pth'%(e))
                



def get_stats(train_loader):
    classes = [0] * 4
    for i, d in enumerate(train_loader):
        print('Getting Stats: %d/%d' %(i, len(train_loader)), end='\r')
        _, a = d
        for j in range(4):
            classes[j] += (a == j).sum().item()

    classes = [round(c * 100, 2) / sum(classes) for c in classes]
    print(classes)

if __name__ == "__main__":
    train(64, 1e-3, 0, 30)