from torch.utils.data import Dataset, DataLoader
from itertools import combinations
import torch
import random

class FeedbackDataset(Dataset):
    def __init__(self, 
                 observations, actions, feedback, 
                 index,
                 window_length=10, window_shift=1, traj_length=1, pair_size=2, 
                 model_type="regression", 
                 no_preference=True, moving_window=True):
        """
        Args:
        observations: Tensor of images (N, C, H, W)
        actions: Tensor of actions (N, A)
        feedback: Tensor of feedback (N, 2)
        window_length: The size of window for sampling traj from
        tarj_length: The size of trajectory to sample
        pair_size: The size of pair to sample, can be a list of integers
        """
        assert len(observations) == len(actions) == len(feedback)
        assert window_length > traj_length
        
        assert model_type in ["regression", "soft_label", "hard_label", "vote_ref"]
        
        self.observations = observations  # Tensor of images (N, C, H, W)
        self.actions = actions  # Tensor of actions (N, A)
        self.feedback = feedback
        self.pair_size = pair_size
        self.model_type = model_type
        
        self.index = index
        
        self.no_preference = no_preference
        self.moving_window = moving_window
        
        
        if model_type != "regression":

            self.window_length = window_length
            self.traj_length = traj_length
            
            
            self.traj = []
            self.data_list = []

            for obs, act, feed in zip(observations, actions, feedback):
                self.traj.append((obs, act, feed))
            
            self.generate_pairs(pair_size, window_length, window_shift)
        else:
            #for regression dataset directly sample from the index
            # print("Regression dataset sample from index")
            self.observations = self.observations[self.index]
            self.actions = self.actions[self.index]
            self.feedback = self.feedback[self.index]

    
    def generate_pairs(self, pair_size, window_length, window_shift):
        
        
        for i in range(0, len(self.traj) - window_length + 1, window_shift):
            if i == 0:
                index_in_window = list(range(i, i + window_length))
                combinations_list = list(combinations(index_in_window, pair_size))
            else:
                if i + window_length >= len(self.traj) - len(self.traj)%window_shift:
                    middle_index = i + window_length - window_shift
                    final_index = len(self.traj)
                                    
                else:
                    middle_index = i + window_length - window_shift
                    final_index = i + window_length
            
                first_half = list(range(i, middle_index))
                temp_combination_list = list(combinations(first_half, pair_size-1))
                combinations_list = []
                
                for comb in temp_combination_list:
                    comb = list(comb)
                    for later_index in range(middle_index, final_index):
                        new_comb = comb.copy()
                        new_comb.append(later_index)
                        
                        combinations_list.append(new_comb)
        
                combinations_list += list(combinations(range(middle_index, final_index), pair_size))
            
    
            for comb1 in combinations_list:
                if comb1[0] in self.index and comb1[1] in self.index:    
                    self.data_list.append([self.traj[i] for i in comb1])
        
        if not self.moving_window:
            old_data_list_len = len(self.data_list)
            
            combinations_list = list(combinations(self.index, pair_size))
            
            if old_data_list_len <= len(combinations_list):
                sample_pairs = random.sample(combinations_list, old_data_list_len)
            else:
                raise ValueError("Not enough unique pairs to sample without replacement.")
        
            self.data_list = []
        
            for comb1 in sample_pairs:
                if comb1[0] in self.index and comb1[1] in self.index:    
                    self.data_list.append([self.traj[i] for i in comb1])
                else:
                    print("Something wrong with the code")
        
        
        
    def __len__(self):
        if self.model_type == "regression":
            return len(self.observations)
        else:
            return len(self.data_list)
    
    def __getitem__(self, idx):
        
        if self.model_type == "regression":    
            return self.observations[idx], self.actions[idx], self.feedback[idx]
        
        else:
            data = self.data_list[idx]
            
            obs_list = []
            act_list = []
            feed_list = []
            
            for traj_data in data:
                obs, act, feed = traj_data

                obs_list.append(obs.squeeze())
                act_list.append(act)
                feed_list.append(feed)
            
            obs = torch.stack(obs_list)
            act = torch.stack(act_list)
            feed = torch.stack(feed_list)
            
            label = torch.zeros_like(feed)
            
            if self.no_preference:
                if torch.abs(feed[0,:,:] - feed[1,:,:]) < 0.05:
                    label[:,:,:] = 0.5
                
                else:
                    if feed[0,:,:] > feed[1,:,:]:
                        max_index = 0
                    else:
                        max_index = 1
                    
                    if self.model_type == "soft_label":
                        label[max_index,:,:] = 0.9
                        label[(max_index+1)%2,:,:] = 0.1
                        
                    
                    elif self.model_type == "hard_label":
                        label[max_index,:,:] = 1
            else:
                if torch.abs(feed[0,:,:] - feed[1,:,:]) == 0:
                    label[:,:,:] = 0.5
                
                else:
                    if feed[0,:,:] > feed[1,:,:]:
                        max_index = 0
                    else:
                        max_index = 1
                    
                    if self.model_type == "soft_label":
                        label[max_index,:,:] = 0.9
                        label[(max_index+1)%2,:,:] = 0.1
                        
                    
                    elif self.model_type == "hard_label":
                        label[max_index,:,:] = 1
            
            return obs, act, label.squeeze()
        


class FeedbackDataset_vote(Dataset):
    def __init__(self, 
                 observations, actions, feedback, 
                 index,
                 window_length=10, window_shift=1, traj_length=1, pair_size=2):
        """
        Args:
        observations: Tensor of images (N, C, H, W)
        actions: Tensor of actions (N, A)
        feedback: Tensor of feedback (N, 2)
        window_length: The size of window for sampling traj from
        tarj_length: The size of trajectory to sample
        pair_size: The size of pair to sample, can be a list of integers
        """
        assert len(observations) == len(actions) == feedback.shape[1]
        assert window_length > traj_length
        
        
        self.observations = observations  # Tensor of images (N, C, H, W)
        self.actions = actions  # Tensor of actions (N, A)
        self.pair_size = pair_size

        self.window_length = window_length
        self.traj_length = traj_length
        
        self.index = index
        
        
        self.traj = []
        self.data_list = []

        for obs, act, feed in zip(observations, actions, feedback.T):
            self.traj.append((obs, act, feed))
        
        self.generate_pairs(pair_size, window_length, window_shift)

    
    def generate_pairs(self, pair_size, window_length, window_shift):
        
        
        for i in range(0, len(self.traj) - window_length + 1, window_shift):
            if i == 0:
                index_in_window = list(range(i, i + window_length))
                combinations_list = list(combinations(index_in_window, pair_size))
            else:
                if i + window_length >= len(self.traj) - len(self.traj)%window_shift:
                    middle_index = i + window_length - window_shift
                    final_index = len(self.traj)
                                    
                else:
                    middle_index = i + window_length - window_shift
                    final_index = i + window_length
            
                first_half = list(range(i, middle_index))
                temp_combination_list = list(combinations(first_half, pair_size-1))
                combinations_list = []
                
                for comb in temp_combination_list:
                    comb = list(comb)
                    for later_index in range(middle_index, final_index):
                        new_comb = comb.copy()
                        new_comb.append(later_index)
                        
                        combinations_list.append(new_comb)
        
                combinations_list += list(combinations(range(middle_index, final_index), pair_size))
            
            for comb1 in combinations_list:
                if comb1[0] in self.index and comb1[1] in self.index:
                    self.data_list.append([self.traj[i] for i in comb1])
        
    def __len__(self):

        return len(self.data_list)
    
    def __getitem__(self, idx):

        data = self.data_list[idx]
        
        obs_list = []
        act_list = []
        feed_list = []
        
        for traj_data in data:
            obs, act, feed = traj_data

            obs_list.append(obs.squeeze())
            act_list.append(act)
            feed_list.append(feed)
        
        obs = torch.stack(obs_list)
        act = torch.stack(act_list)
        feed = torch.stack(feed_list)
        
        label = torch.zeros(size=(2,1))
        
        label = count_ranking(label,feed)
        return obs, act, label
        


def count_ranking(label, feed):
    
    difference = feed[0, :] - feed[1,:]
    for diff in difference:
        if abs(diff) <= 0.05:
            label[0,:] += 1
            label[1,:] += 1
        elif diff > 0.05:
            label[0,:] += 2
        else:  # diff < -0.05
            label[1,:] += 2
        
    
    label = label / label.sum()
    
    return label