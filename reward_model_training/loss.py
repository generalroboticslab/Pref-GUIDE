import torch
import torch.nn as nn
import torch.nn.functional as F
from math import perm
from itertools import permutations

def softmax_preference_loss(reward_tensor, feed, config, reference_reward):
    """
    Computes the preference-based loss using a softmax over rewards.

    Args:
    reward_list: List of rewards for each trajectory
    mu: Tensor of human preference probabilities (batch_size, 2)

    Returns:
    Loss value (scalar)
    """
    assert len(reward_tensor) == feed.size(0), "Reward list and mu size mismatch"
    # Compute softmax probabilities
        
    if reference_reward is not None:
        mu =  torch.zeros(size=(feed.size(0),perm(feed.size(1),feed.size(1)))).to(feed.device)

        rank_count = count_ranking(reference_reward,config)
        for i in range(len(mu)):
            if config.reward_type == "hard_label":
                if (rank_count[i] == reference_reward.size(-1) / perm(feed.size(1),feed.size(1))).prod().item():
                    mu[i] = 1 / perm(feed.size(1),feed.size(1))
                else:
                    mu[i,torch.argmax(rank_count,dim=1)[i]] = 1
            elif config.reward_type == "soft_label":
                if (rank_count[i] == reference_reward.size(-1) / perm(feed.size(1),feed.size(1))).prod().item():
                    mu[i] = 1 / perm(feed.size(1),feed.size(1))
                else:
                    mu[i,torch.argmax(rank_count,dim=1)[i]] = 1 - config.soft_label_alpha
                    for j in range(len(mu[i])):
                        if j != torch.argmax(rank_count,dim=1)[i]:
                            mu[i,j] = config.soft_label_alpha/(len(mu[i])-1)
            elif config.reward_type == "reference_model_vote":
                mu[i] = rank_count[i]/rank_count[i].sum()
        
        
    else:
        if config.reward_type == "hard_label":
            mu = torch.zeros_like(feed)
            for i in range(len(mu)):
                mu[i,torch.argmax(feed,dim=1)[i]] = 1

        elif config.reward_type == "soft_label":
            mu = torch.zeros_like(feed)
            for i in range(len(mu)):
                mu[i,torch.argmax(feed,dim=1)[i]] = 1 - config.soft_label_alpha
                for j in range(len(mu[i])):
                    if j != torch.argmax(feed,dim=1)[i]:
                        mu[i,j] = config.soft_label_alpha/(len(mu[i])-1)
                
        if config.apply_no_preference:
            for i in range(len(mu)):
                if torch.max(feed[i]) - torch.min(feed[i]) <= config.no_preference_range:
                    for j in range(len(mu[i])):
                        mu[i,j] = 1 / feed.size(1)
    prb_list = torch.zeros_like(reward_tensor)
    
    for i in range(len(reward_tensor)):
        prb_list[i] = torch.nn.functional.softmax(reward_tensor[i], dim=0)

    # Compute loss using cross-entropy
    loss = torch.zeros(size= (len(reward_tensor),1))
    for i in range(len(prb_list)):
        loss[i] = -( torch.log(prb_list[i].unsqueeze(0)) @ mu[i].unsqueeze(-1))

    return loss.mean()


def count_ranking(reference_tensor,config):
    ranking_count = torch.zeros(size=(reference_tensor.size(0),perm(reference_tensor.size(1),reference_tensor.size(1))))
    
    permutation_list = list(permutations(range(reference_tensor.size(1)),reference_tensor.size(1)))
    
    if (not config.level_one) and config.apply_no_preference:
            for i in range(len(reference_tensor)):
                for j in range(len(reference_tensor[i].T)):
                    if torch.max(reference_tensor[i,:,j]) - torch.min(reference_tensor[i,:,j]) <= config.no_preference_range:
                        for k in range(len(permutation_list)):
                            ranking_count[i,k] += 1
                        
                    else:
                        index_sorted = tuple(torch.argsort(reference_tensor[i,:,j], descending=True).tolist())
                        ranking_count[i,permutation_list.index(index_sorted)] += reference_tensor.shape[1]
    else:
        for i in range(len(reference_tensor)):
            for j in range(len(reference_tensor[i].T)):
                index_sorted = tuple(torch.argsort(reference_tensor[i,:,j], descending=True).tolist())
                ranking_count[i,permutation_list.index(index_sorted)] += 1
    
    return ranking_count


def softmax_preference_loss_vote_ref(reward_tensor, feed, config, reference_reward):
    """
    Computes the preference-based loss using a softmax over rewards.

    Args:
    reward_list: List of rewards for each trajectory
    mu: Tensor of human preference probabilities (batch_size, 2)

    Returns:
    Loss value (scalar)
    """
    assert len(reward_tensor) == feed.size(0), "Reward list and mu size mismatch"
    # Compute softmax probabilities

    mu =  torch.zeros(size=(feed.size(0),perm(feed.size(1),feed.size(1)))).to(feed.device)

    rank_count = count_ranking(reference_reward,config)
    for i in range(len(mu)):
        mu[i] = rank_count[i]/rank_count[i].sum()
    

    prb_list = torch.zeros_like(reward_tensor)
    
    for i in range(len(reward_tensor)):
        prb_list[i] = torch.nn.functional.softmax(reward_tensor[i], dim=0)

    # Compute loss using cross-entropy
    loss = torch.zeros(size= (len(reward_tensor),1))
    for i in range(len(prb_list)):
        loss[i] = -( torch.log(prb_list[i].unsqueeze(0)) @ mu[i].unsqueeze(-1))

    return loss.mean()
