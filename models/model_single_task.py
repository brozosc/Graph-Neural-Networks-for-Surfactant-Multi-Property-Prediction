# -*- coding: utf-8 -*-
"""
Author: Christoforos Brozos
The python script is developed to train a GNN for CMC or/ Γm predictions. 
The architecture used here was found to be the optimum through hyperparameter tuning.
"""

from smiles_to_graphs import OwnDataset
import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_scatter import scatter_add
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import argparse
import time
import pandas as pd 
from utils import EarlyStopping

# hyperparameters of the model
parser = argparse.ArgumentParser()
parser.add_argument('--plot', default = True) # Plot of train, validation and test error plots
parser.add_argument('--epochs', default=300)   # number of epochs
parser.add_argument('--dim', default=64)   # size of hidden node states
parser.add_argument('--lrate', default=0.005)   #  learning rate
parser.add_argument('--batch', default = 16)  # batch size
parser.add_argument('--early_stopping_patience', default=50)   # number of epochs until early stopping
parser.add_argument('--lrfactor', default=0.8)   # decreasing factor for learning rate
parser.add_argument('--lrpatience', default=3)   # number of consecutive epochs without model improvement after which learning rate is decreased


args = parser.parse_args()
plot = args.plot
epochs = int(args.epochs)
dim = int(args.dim)
lrate = float(args.lrate)
batch = int(args.batch)
lrfactor = float(args.lrfactor)
lrpatience = int(args.lrpatience)
early_stopping_patience = int(args.early_stopping_patience)

# Model geometry

class GNN_CMC(torch.nn.Module):
    def __init__(self):
        super(GNN_CMC, self).__init__()      
        self.lin0 = Linear(dataset.num_features, dim)
        self.gru = GRU(dim,dim)
        
        edge_nn = Sequential(Linear(dataset.num_edge_features, dim), ReLU(), Linear(dim, dim * dim))
        self.conv1 = NNConv(dim, dim, edge_nn,  aggr='add')            
        
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, dim)
        self.fc3 = torch.nn.Linear(dim , 1)

    
    def forward(self, data):
        x, edge_index, edge_attr = data.x , data.edge_index, data.edge_attr
        x = F.relu(self.lin0(data.x))

        h = x.unsqueeze(0)
       
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x,h = self.gru(x.unsqueeze(0),h)
        x = x.squeeze(0)
                       
        x_forward = x
        x_1 = scatter_add(x_forward, data.batch, dim=0)
        x_1 = F.relu(self.fc1(x_1))
        x_1 = F.relu(self.fc2(x_1))
        x_1 = self.fc3(x_1)
        return x_1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the task's corresponding dataset

dataset = OwnDataset(root = "")
ext_test_dataset = OwnDataset(root = "")
  
# Normalize the target property to mean=0 and std =1    
mean = torch.as_tensor(dataset.data.y, dtype=torch.float).mean()
std = torch.as_tensor(dataset.data.y, dtype=torch.float).std()
dataset.data.y = (dataset.data.y - mean) / std
ext_test_dataset.data.y = (ext_test_dataset.data.y - mean) / std
print('Model parameters: ' + str(sum(p.numel() for p in GNN_CMC().parameters())))
print('Target data mean: ' + str(mean.tolist()))
print('Target data standard deviation: ' + str(std.tolist()))

# Split the training dataset into training and validation sets. The split is different in every reputation based on the seed
# to ensure model robustness. The test dataset remains always the same.

def data_preparation(seed,dataset = dataset):
    torch.manual_seed(seed)
    dataset = dataset.shuffle()
    val_dataset = dataset[:20]
    train_dataset = dataset[20:]
    train_loader = DataLoader(train_dataset, batch_size = batch, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = len(val_dataset))
    test_loader = DataLoader(ext_test_dataset[:], batch_size = len(ext_test_dataset))
    return train_loader, val_loader, test_loader

def train(loader, model, optimizer):
    model.train()
    std_pred_error = 0
    loss_all = abs_loss_all = total_examples = 0
    norm_train_mae, train_mae = 0, 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        real_data = data.y
        loss = F.mse_loss(out, real_data)
        loss.backward()
        loss_all += loss * data.num_graphs
        total_examples += data.num_graphs
        optimizer.step()
        
        #Normalized errors calculation
        norm_train_rmse = torch.sqrt(loss_all/total_examples)
        norm_train_mae += (out - real_data).abs().sum(0).item()  
        
        #counting the de-normalized errors
        out_denormalized = out*std + mean
        real_data_denormalized = real_data*std + mean
        abs_loss = F.mse_loss(out_denormalized, real_data_denormalized)
        abs_loss_all += abs_loss*data.num_graphs
        train_rmse = torch.sqrt(abs_loss_all/total_examples)     
        train_mae += ((out - real_data)/(real_data+mean/std)).abs().sum(0).item()
        
     #We report only de-normalized errors   
    return loss_all / len(loader.dataset), train_rmse.item(), train_mae/ len(loader.dataset)

def test(loader, model, optimizer):
    model.eval()
    val_mae, val_rmse = 0, 0
    loss_all_norm = norm_val_mae = abs_loss_all = 0 
    loss_all = total_examples = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            real_data = data.y
            loss = F.mse_loss(out, real_data)
            loss_all += loss * data.num_graphs
            total_examples += data.num_graphs
        
            #calculating normalized errors
            norm_val_rmse = torch.sqrt(loss_all/total_examples)
            norm_val_mae += (out - real_data).abs().sum(0).item()
        
            #counting the de-normalized errors
            out_denormalized = out*std + mean
            real_data_denormalized = real_data*std + mean
            abs_loss = F.mse_loss(out_denormalized, real_data_denormalized)
            abs_loss_all += abs_loss*data.num_graphs
            val_rmse = torch.sqrt(abs_loss_all/total_examples) 
            val_mae += (out_denormalized - real_data_denormalized).abs().sum(0).item()
          
    #We report only de-normalized errors
    return val_rmse.item(), val_mae / len(loader.dataset)


#User defined path for model saving and loading
save_path = str('')

def save_checkpoint(model,filename):
    print('Saving checkpoint') 
    torch.save(model.state_dict(), save_path+filename)    
    time.sleep(2)


# This function create different training-validation splits, initiates the model training and saves the model on the best epoch.
def training(counter):
    best_epoch, best_val_rmse , best_epoch_test_rmse = None, None, None
    best_val_mae, best_epoch_test_mae = None, None
    train_errors, val_errors, test_errors = [], [],[]
    train_loader, validation_loader, test_loader = data_preparation(seed=counter)
    
    model = GNN_CMC().to(device)
    print(model)    
    
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = lrate)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lrfactor, patience=lrpatience, min_lr=0.0000001)
     
    for epoch in range(1,epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss, train_rmse, train_mae = train(train_loader, model, optimizer)
        
        val_rmse, val_mae = test(validation_loader, model, optimizer)
        
        test_rmse, test_mae = test(test_loader, model, optimizer)
        print('Epoch: {} , Learning Rate: {}, Val: val_rmse {:.5f}, val_mae {:.5f}, Test: test_rmse {:.5f}, test_mae {:.5f}'.format(epoch, lr, val_rmse, val_mae, test_rmse, test_mae))
        
        scheduler.step(val_rmse)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
        test_errors.append(test_rmse)
        
        
        if best_val_rmse is None:
            best_epoch = epoch
            best_val_rmse, best_epoch_test_rmse = val_rmse, test_rmse
            best_val_mae, best_epoch_test_mae = val_mae, test_mae
        elif val_rmse < best_val_rmse:
            best_epoch = epoch
            best_val_rmse, best_epoch_test_rmse = val_rmse, test_rmse
            best_val_mae, best_epoch_test_mae = val_mae, test_mae
            save_checkpoint(model,'base_model_{}.pt'.format(counter))
        
        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print(
    'Best model with respect to validation error in epoch {:03d} with \nVal RMSE {:.5f}\nTest RMSE {:.5f}\n'
    .format(best_epoch, best_val_rmse, best_epoch_test_rmse))  
    

    if plot is True:
         plt.plot(range(1,len(train_errors)+1), train_errors, label = 'Train')
         plt.plot(range(1,len(val_errors)+1), val_errors, label = 'Validation')
         plt.plot(range(1,len(test_errors)+1), test_errors, label='Test')
         plt.xlabel('Epochs')
         plt.ylabel('RMSE')
         plt.grid(True)
         plt.legend(frameon=False)
         plt.tight_layout()
         plt.show()
    else:
        pass
    return loss, best_val_rmse, best_epoch_test_rmse, best_val_mae, best_epoch_test_mae


single_seed_results = training(1)
validation_set_rmse = single_seed_results[1]
validation_set_mae = single_seed_results[3]
test_set_rmse = single_seed_results[2]
test_set_mae = single_seed_results[4]

