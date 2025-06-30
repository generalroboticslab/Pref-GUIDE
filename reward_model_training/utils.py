import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import wandb
import tqdm
import torch
import os
from torch.utils.data import DataLoader
from dataloader import FeedbackDataset , FeedbackDataset_vote
from models import Feedback_Net, Encoder
from torch.utils.data import ConcatDataset

def plot_feedback(feedback,name):
    feedback = np.array(feedback)
    plt.figure(figsize=(80, 10))  # Set figure size
    plt.plot(feedback)
    plt.ylabel('Feedback')
    plt.xlabel('Step')
    plt.savefig(f'{name}.png', bbox_inches='tight')  # Save with tight layout
    plt.close()  # Close the figure to free memory

def save_image1(sorted_images):
    for i in range (len(sorted_images)):
        save_image(sorted_images[i,:,-3:,:,:].squeeze(), f'Frame_{i}.png')
        

def plot_comparison(ref_feedback, simulated_feedback, name):
    ref_feedback = np.array(ref_feedback)
    simulated_feedback = np.array(simulated_feedback)
    
    plt.figure(figsize=(80, 10))  # Set figure size
    plt.plot(ref_feedback, label='Reference Feedback', color='blue')
    plt.plot(simulated_feedback, label='Simulated Feedback', color='red')
    plt.ylabel('Feedback')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig(f'{name}.png', bbox_inches='tight')  # Save with tight layout
    plt.close()  # Close the figure to free memory

def train_model(model, config, trainLoader, validationLoader, device, optimizer, criterion):
    
    pretrain = "_pretrain" if config.pretrain else ""
    if config.freeze_encoder:
        pretrain = "_freeze"
    
    activation = "_activation" if config.use_activation else ""
    
    
    no_preference = "_no_random_window" if not config.no_preference_window else ""
    no_moving_window = "_no_moving_window" if not config.moving_window else ""
    
    exp_name = f"{config.game}_{config.model_type}_{pretrain}_subject_{config.subject_id}{activation}{no_preference}{no_moving_window}"
    
    activation_tag = "activation" if config.use_activation else "no_activation"

    wandb.login()
    wandb.init(
        project="Pref-GUIDE_reward_model_training-CREW", 
        config=config, 
        name = exp_name,
        tags=[config.game, config.model_type, f"subject_{config.subject_id}", activation_tag]
    )
    
    best_validation_loss = float('inf')
    training_loss = float('inf')

    pbar = tqdm.tqdm(range(config.num_epochs * len(trainLoader)))

    for epoch in range(config.num_epochs):
        
        total_training_loss = 0
        
        total_accurate_train = 0
        total_count_train = 0
            

        for i , (img,action,mu) in enumerate(trainLoader):
            img, action, mu= img.to(device), action.to(device), mu.to(device)
            optimizer.zero_grad()
            if config.model_type == "regression":
                output = model(img,action)
            else:
                img1,img2 = img[:,0,:,:,:].unsqueeze(1),img[:,1,:,:,:].unsqueeze(1)
                output1 = model(img1 , action[:,0,:,:,:])
                output2 = model(img2 , action[:,1,:,:,:])
                
                output = torch.cat((output1,output2),dim=1).squeeze()
                output = torch.softmax(output,dim=1)      
            try:
                loss = criterion(output, mu)
            except:
                loss = criterion(output, mu.squeeze())
            loss.backward()
            optimizer.step()
            
            total_training_loss += loss.mean().item()
            
            if config.model_type != "regression":
                diff = output[:,0] - output[:,1]
                diff = torch.abs(diff)

                predicted_labels = torch.where(
                    diff < 0.1,
                    torch.tensor(0, device=diff.device),
                    torch.where(
                        output[:,0]  > output[:,1],
                        torch.tensor(1, device=diff.device),
                        torch.tensor(2, device=diff.device)
                    )
                )
                
                predicted_labels1 = torch.where(
                    output[:,0] > output[:,1],
                    torch.tensor(1, device=diff.device),
                    torch.tensor(2, device=diff.device)
                )
                
                true_labels = torch.where(
                    mu[:,0] == mu[:,1],
                    torch.tensor(0, device=diff.device),
                    torch.where(
                        mu[:,0]  > mu[:,1],
                        torch.tensor(1, device=diff.device),
                        torch.tensor(2, device=diff.device)
                    )
                ).squeeze()
            
                total_accurate_train += ((true_labels == predicted_labels) | (true_labels == predicted_labels1)).sum().item()

                total_count_train += img.size(0)
            
            
            pbar.update(1)
            
            
        training_loss = total_training_loss / len(trainLoader)
        
        if config.model_type != "regression":
            training_accuracy = total_accurate_train / total_count_train
        
        with torch.no_grad():
            model.eval()
            
            total_validation_loss = 0

            if config.model_type != "regression":
                total_count = 0
                total_accurate = 0
            
            for img,action,mu in validationLoader:
                img, action, mu = img.to(device), action.to(device), mu.to(device)
                if config.model_type == "regression":
                    output = model(img,action)
                else:
                    img1,img2 = img[:,0,:,:,:].unsqueeze(1),img[:,1,:,:,:].unsqueeze(1)
                    output1 = model(img1 , action[:,0,:,:,:])
                    output2 = model(img2 , action[:,1,:,:,:])
                    
                    output = torch.cat((output1,output2),dim=1).squeeze()
                    output = torch.softmax(output,dim=1)
                
                try:
                    total_validation_loss += criterion(output, mu).item()
                except:
                    total_validation_loss += criterion(output, mu.squeeze()).item()
                
                if config.model_type != "regression":
                    diff = output[:,0] - output[:,1]
                    diff = torch.abs(diff)

                    predicted_labels = torch.where(
                        diff < 0.1,
                        torch.tensor(0, device=diff.device),
                        torch.where(
                            output[:,0]  > output[:,1],
                            torch.tensor(1, device=diff.device),
                            torch.tensor(2, device=diff.device)
                        )
                    )
                    
                    predicted_labels1 = torch.where(

                        output[:,0]  > output[:,1],
                        torch.tensor(1, device=diff.device),
                        torch.tensor(2, device=diff.device)

                    )
                    
                    true_labels = torch.where(
                        mu[:,0] == mu[:,1],
                        torch.tensor(0, device=diff.device),
                        torch.where(
                            mu[:,0]  > mu[:,1],
                            torch.tensor(1, device=diff.device),
                            torch.tensor(2, device=diff.device)
                        )
                    ).squeeze()
                    
                    # print(predicted_labels.shape)
                    # print(true_labels.shape)
                    
                    # print((true_labels == predicted_labels).sum())
                
                    total_accurate +=  ((true_labels == predicted_labels) | (true_labels == predicted_labels1)).sum().item()
                    total_count += img.size(0)
                    # print(total_accurate)
                    # print(total_count)
                    # breakpoint()
            
            validation_loss = total_validation_loss / len(validationLoader)
            
            if config.model_type != "regression":
                accuracy = total_accurate / total_count
            
            if config.model_type == "regression":
                wandb.log({"Training Loss": training_loss, "Validation Loss": validation_loss})
            else:
                wandb.log({"Training Loss": training_loss, "Training Accuracy": training_accuracy,  "Validation Loss": validation_loss, "Validation Accuracy": accuracy})
            
            
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                
                saved_path = f"../reward_model/{config.model_type}{pretrain}{activation}{no_preference}{no_moving_window}/{config.game}/"
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                
                torch.save(model.state_dict(), f"{saved_path}/subject_{config.subject_id}.pt")
                
                best_model = model.state_dict()
                
                last_epoch_saved = epoch

            if config.model_type == "regression":
                pbar.set_description(f"Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, Last Epoch Saved: {last_epoch_saved}")
            else:
                pbar.set_description(f"Training Loss: {training_loss:.4f}, training Accuracy: {training_accuracy:.4f}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.4f}, Last Epoch Saved: {last_epoch_saved}")
            pbar.refresh()
            
            model.train()
        
    wandb.finish()
    
    return best_model
    

def load_data(config):

    # Load data
    id = str(config.subject_id) if config.subject_id >= 10 else "0" + str(config.subject_id)

    game_name = config.game
    
    image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
    action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
    feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_human_feedback.pt")
    traj_ids = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_tarj_ids.pt")
    
    train_index = []
    validation_index = []
    
    for i in range(len(traj_ids)):
        if traj_ids[i] % 10 == 9:
            validation_index.append(i)
        else:
            train_index.append(i) 

        

    trainDataset = FeedbackDataset(image_tensor, action_tensor, feedback_tensor, 
                                   train_index,
                                   window_length=10, window_shift=1, traj_length=1, pair_size=2, 
                                   model_type=config.model_type,
                                   no_preference=config.no_preference_window, moving_window=config.moving_window
                                   )        
    validationDataset = FeedbackDataset(image_tensor, action_tensor, feedback_tensor, 
                                        validation_index,
                                        window_length=10, window_shift=1, traj_length=1, pair_size=2, 
                                        model_type=config.model_type,
                                        no_preference=config.no_preference_window, moving_window=config.moving_window
                                        )   

    # print(validation_index)

    print(f"Subect {config.subject_id}, Training Size: {len(trainDataset)}, Validation Size: {len(validationDataset)}")


    # create DataLoader
    batch_size = config.batch_size
    num_workers = 64    

    trainLoader = DataLoader(trainDataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    validationLoader = DataLoader(validationDataset, batch_size = batch_size, shuffle=False, num_workers=num_workers)
    
    return trainLoader, validationLoader, train_index, validation_index


def make_model(config, device):
    #optimizer and Model
    encoder = Encoder(in_channels=3, embedding_dim=64)

    if config.pretrain:
        sub_id = f"0{config.subject_id}" if config.subject_id < 10 else f"{config.subject_id}"
        try:
            model_weight = torch.load(f"../../guide_data/{sub_id}_data/saved_training/{sub_id}_{config.game}_guide/weights_Iter_5.pth")
        except:
            model_weight = torch.load(f"../../guide_data/{sub_id}_Data/saved_training/{sub_id}_{config.game}_guide/weights_Iter_5.pth")
        encoder_weight = {}
        for key in model_weight.keys():
            if "encoder" in key:
                new_key = key.split("encoder.")[-1]
                encoder_weight[new_key] = model_weight[key]
        
        encoder.load_state_dict(encoder_weight)
        print("Pretrain model loaded")
        
        if config.freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen")
        
        
    model = Feedback_Net(encoder=encoder, n_agent_inputs=2 + 64*3, num_cells=256, activation_class=torch.nn.ReLU, use_activation=config.use_activation)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(model)

    #define Loss
    if config.model_type == "regression":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, criterion

def eval_model(model, config, train_index, device):
    # Load data
    model.eval()
    
    id = str(config.subject_id) if config.subject_id >= 10 else "0" + str(config.subject_id)

    game_name = config.game
    
    image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
    action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
    feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_human_feedback.pt")
    
    image_tensor = image_tensor.to(device)
    action_tensor = action_tensor.to(device)
    
    color_list = []
    for i in range(len(image_tensor)):
        if i in train_index:
            color = "blue"
        else:
            color = "orange"
        color_list.append(color)
    
    simulated_feedback = model(image_tensor, action_tensor)
    simulated_feedback = simulated_feedback.squeeze().cpu().detach().numpy()
    feedback_tensor = feedback_tensor.squeeze().cpu().detach().numpy()
    
    pretrain = "_pretrain" if config.pretrain else ""
    if config.freeze_encoder:
        pretrain = "_freeze"
    
    if config.use_activation:
        activation_tag = "_activation"
    else:
        activation_tag = ""
        
    no_preference = "_no_random_window" if not config.no_preference_window else ""
    no_moving_window = "_no_moving_window" if not config.moving_window else ""
    
    folder = f"../reward_model/{config.model_type}{pretrain}{activation_tag}{no_preference}{no_moving_window}/{config.game}/figures/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    plt.figure(figsize=(80, 10))  # Set figure size
    plt.plot(feedback_tensor, label='Reference Feedback')
    plt.plot(simulated_feedback, label='Simulated Feedback')
    plt.ylabel('Feedback')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig(f'{folder}/{config.subject_id}.png', bbox_inches='tight')  # Save with tight layout
    plt.close()  # Close the figure to free memory
    
def load_data_vote_ref(config):

    # Load data
    id = str(config.subject_id) if config.subject_id >= 10 else "0" + str(config.subject_id)

    game_name = config.game
    
    image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_img.pt")
    action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_action.pt")
    feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_simulated_reward_{config.ref_model}.pt")
    traj_ids = torch.load(f"../../Preference_guide_Data/{game_name}/{id}/{id}_tarj_ids.pt")
    
    train_index = []
    validation_index = []
    
    for i in range(len(traj_ids)):
        if traj_ids[i] % 10 == 9:
            validation_index.append(i)
        else:
            train_index.append(i) 

    # create DataLoader
    
    trainDataset = FeedbackDataset_vote(image_tensor, action_tensor, feedback_tensor, 
                                   train_index,
                                   window_length=10, window_shift=1, traj_length=1, pair_size=2)        
    validationDataset = FeedbackDataset_vote(image_tensor, action_tensor, feedback_tensor, 
                                        validation_index,
                                        window_length=10, window_shift=1, traj_length=1, pair_size=2)   

    # print(validation_index)

    print(f"Subect {config.subject_id}, Training Size: {len(trainDataset)}, Validation Size: {len(validationDataset)}")


    # create DataLoader
    batch_size = config.batch_size
    num_workers = 64    

    trainLoader = DataLoader(trainDataset, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    validationLoader = DataLoader(validationDataset, batch_size = batch_size, shuffle=False, num_workers=num_workers)
    
    return trainLoader, validationLoader, train_index, validation_index


def load_data_vote_ref_total(config):
    
    all_train_datasets = []
    all_validation_datasets = []

    for id in range(52):
        if id == 12 or id == 42:  # fixed the condition
            continue
        id_str = str(id) if id >= 10 else "0" + str(id)

        game_name = config.game
        
        image_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id_str}/{id_str}_img.pt")
        action_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id_str}/{id_str}_action.pt")
        feedback_tensor = torch.load(f"../../Preference_guide_Data/{game_name}/{id_str}/{id_str}_simulated_reward_{config.ref_model}.pt")
        traj_ids = torch.load(f"../../Preference_guide_Data/{game_name}/{id_str}/{id_str}_tarj_ids.pt")

        train_index = []
        validation_index = []

        for i in range(len(traj_ids)):
            if traj_ids[i] % 10 == 9:
                validation_index.append(i)
            else:
                train_index.append(i)

        trainDataset = FeedbackDataset_vote(
            image_tensor, action_tensor, feedback_tensor, 
            train_index,
            window_length=10, window_shift=1, traj_length=1, pair_size=2
        )
        validationDataset = FeedbackDataset_vote(
            image_tensor, action_tensor, feedback_tensor, 
            validation_index,
            window_length=10, window_shift=1, traj_length=1, pair_size=2
        )

        all_train_datasets.append(trainDataset)
        all_validation_datasets.append(validationDataset)

    # Combine all datasets
    combined_train_dataset = ConcatDataset(all_train_datasets)
    combined_validation_dataset = ConcatDataset(all_validation_datasets)

    print(f"Total Training Size: {len(combined_train_dataset)}, Total Validation Size: {len(combined_validation_dataset)}")

    return combined_train_dataset, combined_validation_dataset