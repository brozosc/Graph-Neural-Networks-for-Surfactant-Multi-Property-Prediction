# -*- coding: utf-8 -*-
"""
Author: Christoforos Brozos
The python script is developed to train a GNN for multi-task CMC and Γm predictions. 
The architecture used here was found to be the optimum through hyperparameter tuning.
The script allows user to train different model initiated on different splits.
"""

from smiles_to_graphs_ml import OwnDataset
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
parser.add_argument('--task', default = 'CMC') # define the predicting task
parser.add_argument('--plot', default = True) # Plot of train, validation and test error plots
parser.add_argument('--epochs', default=300)   # number of epochs
parser.add_argument('--dim', default=64)   # size of hidden node states
parser.add_argument('--lrate', default=0.005)   #  learning rate
parser.add_argument('--batch', default = 16)  # batch size
parser.add_argument('--early_stopping_patience', default=50)   # number of epochs until early stopping
parser.add_argument('--lrfactor', default=0.8)   # decreasing factor for learning rate
parser.add_argument('--lrpatience', default=3)   # number of consecutive epochs without model improvement after which learning rate is decreased


args = parser.parse_args()
task = args.task
plot = args.plot
epochs = int(args.epochs)
dim = int(args.dim)
lrate = float(args.lrate)
batch = int(args.batch)
lrfactor = float(args.lrfactor)
lrpatience = int(args.lrpatience)
early_stopping_patience = int(args.early_stopping_patience)


# Define the model geometry. We import flexibility to the code, so that it adapts based on the user's input

class GNN_CMC(torch.nn.Module):
    def __init__(self):
        super(GNN_CMC, self).__init__()
        self.lin0 = Linear(dataset.num_features, dim)
        
        self.gru = GRU(dim,dim)
                   
        edge_nn = Sequential(Linear(dataset.num_edge_features, dim), ReLU(), Linear(dim, dim * dim))
        self.conv1 = NNConv(dim, dim, edge_nn,  aggr='add')
 
        
        self.fc11 = torch.nn.Linear(dim, dim)
        self.fc12 = torch.nn.Linear(dim, dim)
        self.fc13 = torch.nn.Linear(dim , 1)
        
        self.fc21 = torch.nn.Linear(dim, dim)
        self.fc22 = torch.nn.Linear(dim, dim)
        self.fc23 = torch.nn.Linear(dim , 1)
        
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x , data.edge_index, data.edge_attr
        x = F.relu(self.lin0(data.x))
        
        h = x.unsqueeze(0)
     
        x = F.relu(self.conv1(x, edge_index, edge_attr))
      
        x,h = self.gru(x.unsqueeze(0),h)
        x = x.squeeze(0)
                    
                       
        x_forward = x

        x = scatter_add(x_forward, data.batch, dim=0)
        x_1 = F.relu(self.fc11(x))
        x_1 = F.relu(self.fc12(x_1))
        x_1 = self.fc13(x_1)
        
        x_2 = F.relu(self.fc21(x))
        x_2 = F.relu(self.fc22(x_2))
        x_2 = self.fc23(x_2)
        
        x = torch.cat([x_1,x_2], dim = 1)
        
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the task's corresponding dataset

dataset = OwnDataset(root = r"")
ext_test_dataset = OwnDataset(root = r"")



mean_lst = []
std_lst = []

for target in range(0,len(dataset.data.y[0])):
    tmp_target_data = dataset.data.y[0:,target]
    mean_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).mean())
    std_lst.append(torch.as_tensor(tmp_target_data[~torch.isinf(tmp_target_data)]).std())

## Normalize targets to mean = 0 and std = 1.
mean = torch.tensor(mean_lst, dtype=torch.float)
std = torch.tensor(std_lst, dtype=torch.float)
dataset.data.y = (dataset.data.y - mean) / std
ext_test_dataset.data.y = (ext_test_dataset.data.y - mean) / std
print('Model parameters: ' + str(sum(p.numel() for p in GNN_CMC().parameters())))

# Split the training dataset into training and validation sets. The split is different in every reputation based on the seed
# to ensure model robustness. The test dataset remains always the same


def data_preparation(seed):
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
    loss_all = abs_loss_all_0 = abs_loss_all_1 = 0
    total_samples = total_samples_0 = total_samples_1  = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        mask = ~torch.isinf(data.y)
        real_data = data.y[mask].view(-1,1)
        out_mod = out[mask].view(-1,1)
        loss = F.mse_loss(out_mod, real_data)
        loss.backward()
        optimizer.step()
        loss_all += loss * mask.sum().item()         #scaling the loss based on the available data points
        total_samples += mask.sum().item()          # count of the real points
        
        # property 1
        out_mod_0 = out[:,0]
        out_mod_00 = out_mod_0[mask[:,0]]   
        real_data_0 = data.y[:,0]
        real_data_00 = real_data_0[mask[:,0]]

        total_samples_0 += out_mod_00.numel()
        out_denormalized_0 = out_mod_00*std[0] + mean[0]
        real_data_denormalized_0 = real_data_00*std[0] + mean[0]
        
        abs_loss_0 = F.mse_loss(out_denormalized_0, real_data_denormalized_0)
        abs_loss_all_0 += abs_loss_0* mask[:,0].sum().item()  
        
        # relative error for property 2
        
        out_mod_1 = out[:,1]
        out_mod_11 = out_mod_1[mask[:,1]]
        
        real_data_1 = data.y[:,1]
        real_data_11 = real_data_1[mask[:,1]]
        total_samples_1 += out_mod_11.numel()     
        out_denormalized_1 = out_mod_11*std[1] + mean[1]
        real_data_denormalized_1 = real_data_11*std[1] + mean[1] 
        abs_loss_1 = F.mse_loss(out_denormalized_1, real_data_denormalized_1)
        abs_loss_all_1 += abs_loss_1* mask[:,1].sum().item() 
    
    val_rmse_0 = torch.sqrt(abs_loss_all_0/total_samples_0) 
    val_rmse_1 = torch.sqrt(abs_loss_all_1/total_samples_1) 
       
    return loss_all / total_samples, val_rmse_0.item() , val_rmse_1.item()

def test(loader, model, optimizer):
    model.eval()
    val_rmse_0, val_rmse_1 = 0, 0
    mae_loss0, mae_loss1 = 0,0
    abs_loss_all_0 = abs_loss_all_1 = loss_all = 0
    total_samples, total_samples_0, total_samples_1 = 0,0,0
    for data in loader:
        data = data.to(device)
        out = model(data)
        mask = ~torch.isinf(data.y)
        
        real_data = data.y[mask].view(-1,1)
        out_mod = out[mask].view(-1,1)
        loss = F.mse_loss(out_mod, real_data)
        loss_all += loss * mask.sum().item()
        total_samples += mask.sum().item()
        total_val_error = torch.sqrt(loss_all/total_samples)
        
        #predictions of property 1
        
        out_mod_0 = out[:,0]
        out_mod_00 = out_mod_0[mask[:,0]]        
        
        #real data of property 1
        real_data_0 = data.y[:,0]
        real_data_00 = real_data_0[mask[:,0]]
        
        
        #calculating error matrices for property 1
        total_samples_0 += out_mod_00.numel()
        out_denormalized_0 = out_mod_00*std[0] + mean[0]
        real_data_denormalized_0 = real_data_00*std[0] + mean[0]
        abs_loss_0 = F.mse_loss(out_denormalized_0, real_data_denormalized_0)
        abs_loss_all_0 += abs_loss_0* mask[:,0].sum().item()  
        val_rmse_0 = torch.sqrt(abs_loss_all_0/total_samples_0) 
        mae_loss0 += torch.abs(out_denormalized_0 - real_data_denormalized_0).sum(0).item()        
    
        #predictions of property 2
        
        out_mod_1 = out[:,1]
        out_mod_11 = out_mod_1[mask[:,1]]
        
        #real data of property 2
        
        real_data_1 = data.y[:,1]
        real_data_11 = real_data_1[mask[:,1]]
                    
        #calculating error matrices for property 1
        total_samples_1 += out_mod_11.numel()
        out_denormalized_1 = out_mod_11*std[1] + mean[1]
        real_data_denormalized_1 = real_data_11*std[1] + mean[1]
        abs_loss_1 = F.mse_loss(out_denormalized_1, real_data_denormalized_1)
        abs_loss_all_1 += abs_loss_1* mask[:,1].sum().item() 
        val_rmse_1 = torch.sqrt(abs_loss_all_1/total_samples_1) 
        mae_loss1 += torch.abs(out_denormalized_1 - real_data_denormalized_1).sum(0).item()       

    
    return total_val_error.item(), val_rmse_0.item(), mae_loss0/total_samples_0, val_rmse_1.item(), mae_loss1/total_samples_1

# Write predictions on the given dataset in an Excel file, together with the corresponding Smiles string. The results are been printed in an Excel File.

def write_predictions(loader, model, save_path, dataset_type,counter):
    model = model
    model.load_state_dict(torch.load(save_path+'base_model_{}.pt'.format(counter)))
    model.eval()
    smiles, measured_0, measured_1, predicted_0, predicted_1 = [],[],[],[], []
    df_exp = pd.DataFrame()
    mol_id, pred, real_value, mol_names, pred_list, rmse_errors, mae_errors = None, None, None, [], [], [], []
    for data in loader:
        mol_id = data.mol_id.tolist()
        for mol in mol_id:
            tmp_mol_name = ''
            for i in mol:
                if int(i) is not 0:
                    tmp_mol_name += chr(int(i))
            mol_names.append(tmp_mol_name)           

        real_value_0 = data.y[:,0].tolist()
        real_value_1 = data.y[:,1].tolist()
        data = data.to(device)
        pred = model(data).tolist()
           
        for c, k in enumerate(pred):
            pred_list.append([mol_names[c],(real_value_0[c]*std[0] + mean[0]).item(),(real_value_1[c]*std[1] + mean[1]).item(), (k[0]*std[0] + mean[0]).item(), (k[1]*std[1] + mean[1]).item()])
        mol_names = []  

        for k in pred_list:
            smiles.append(k[0])
            measured_0.append(k[1])
            measured_1.append(k[2])
            predicted_0.append(k[3])
            predicted_1.append(k[4])
            
        df_exp['SMILES'] = smiles
        df_exp['Predicted_0'] = predicted_0
        df_exp['Measured_0'] = measured_0
        df_exp['Predicted_1'] = predicted_1
        df_exp['Measured_1'] = measured_1
        df_exp.to_excel(str(save_path)+str(dataset_type)+'_base_model_{}.xlsx'.format(counter), index = False)
        return mae_errors

# User defined path for model saving and loading
save_path = str('')


def save_checkpoint(model,filename):    
    print('Saving checkpoint')
    torch.save(model.state_dict(), save_path+filename)    
    time.sleep(2)


def training(counter):
    best_epoch, best_val_rmse_0, best_epoch_test_rmse_0, best_val_rmse_1, best_epoch_test_rmse_1 = None, None, None, None, None
    best_val_mae_0, best_epoch_test_mae_0, best_val_mae_1, best_epoch_test_mae_1 = None, None, None, None
    train_losses, val_losses, test_losses = [],[],[]
    train_errors_0, train_errors_1 = [],[]
    val_errors_0 = []
    test_errors_0 = []
    val_errors_1 = []
    test_errors_1 = []
    train_loader, validation_loader, test_loader = data_preparation(seed=counter)
    model = GNN_CMC().to(device)
    
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr = lrate)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=lrfactor, patience=lrpatience, min_lr=0.0000001)
     

    for epoch in range(1,epochs + 1):
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss, train_rmse_0, train_rmse_1 = train(train_loader, model, optimizer)
        
        val_error, val_rmse_0, val_mae_0, val_rmse_1, val_mae_1  = test(validation_loader, model, optimizer)
        
        test_error, test_rmse_0, test_mae_0, test_rmse_1, test_mae_1 = test(test_loader, model, optimizer)
        print('Epoch: {} , Learning Rate: {}, Validation error: {:.5f}, Val: val_rmse_0 {:.5f}, val_mae_0 {:.5f}, Test: test_rmse_0 {:.5f}, test_mae_0 {:.5f}'.format(epoch, lr,val_error, val_rmse_0, val_mae_0, test_rmse_0, test_mae_0))      
        print('Epoch: {} , Learning Rate: {}, Val: val_rmse_1 {:.5f}, val_mae_1 {:.5f}, Test: test_rmse_1 {:.5f}, test_mae_1 {:.5f}'.format(epoch, lr, val_rmse_1, val_mae_1, test_rmse_1, test_mae_1))

        scheduler.step(val_error)
        train_losses.append(loss.detach().numpy())
        val_losses.append(val_error)
        test_losses.append(test_error)
        
        train_errors_0.append(train_rmse_0)
        val_errors_0.append(val_rmse_0)
        test_errors_0.append(test_rmse_0)
        
        train_errors_1.append(train_rmse_1)
        val_errors_1.append(val_rmse_1)
        test_errors_1.append(test_rmse_1)
        
        
        
        if best_val_rmse_0 is None:
            best_epoch = epoch
            best_val_rmse_0, best_epoch_test_rmse_0 = val_rmse_0, test_rmse_0
            best_val_mae_0, best_epoch_test_mae_0 = val_mae_0, test_mae_0
            best_val_rmse_1, best_epoch_test_rmse_1 = val_rmse_1, test_rmse_1
            best_val_mae_1, best_epoch_test_mae_1 = val_mae_1, test_mae_1
        elif val_rmse_0 < best_val_rmse_0:
            best_epoch = epoch
            best_val_rmse_0, best_epoch_test_rmse_0 = val_rmse_0, test_rmse_0
            best_val_mae_0, best_epoch_test_mae_0 = val_mae_0, test_mae_0
            best_val_rmse_1, best_epoch_test_rmse_1 = val_rmse_1, test_rmse_1
            best_val_mae_1, best_epoch_test_mae_1 = val_mae_1, test_mae_1
            save_checkpoint(model,'base_model_{}.pt'.format(counter))
        
        early_stopping(val_error)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print(
    'Best model with respect to validation error in epoch {:03d} with \nVal0 RMSE {:.5f}, Test0 RMSE {:.5f}, Val1 RMSE {:.5f} Test1 RMSE {:.5f}'
    .format(best_epoch, best_val_rmse_0, best_epoch_test_rmse_0, best_val_rmse_1, best_epoch_test_rmse_1))  
    

   # write_predictions(test_loader,model, save_path, 'test', counter = counter)
 
   
    if plot is True:
         plt.title('Total Loss')
         plt.plot(range(1,len(train_losses)+1), train_losses, label = 'Train')
         plt.plot(range(1,len(val_losses)+1), val_losses, label = 'Validation')
         plt.plot(range(1,len(test_losses)+1), test_losses, label='Test')
         plt.xlabel('Epochs')
         plt.ylabel('MSE')
         plt.grid(True)
         plt.legend(frameon=False)
         plt.tight_layout()
         plt.show()
         
         
         plt.title('Property 1')
         plt.plot(range(1,len(train_errors_0)+1), train_errors_0, label = 'Train')
         plt.plot(range(1,len(val_errors_0)+1), val_errors_0, label = 'Validation')
         plt.plot(range(1,len(test_errors_0)+1), test_errors_0 , label='Test')
         plt.xlabel('Epochs')
         plt.ylabel('RMSE')
         plt.grid(True)
         plt.legend(frameon=False)
         plt.tight_layout()
         plt.show()
         
         plt.title('Property 2')
         plt.plot(range(1,len(train_errors_1)+1), train_errors_1, label = 'Train')
         plt.plot(range(1,len(val_errors_1)+1), val_errors_1, label = 'Validation')
         plt.plot(range(1,len(test_errors_1)+1), test_errors_1 , label='Test')
         plt.xlabel('Epochs')
         plt.ylabel('RMSE')
         plt.grid(True)
         plt.legend(frameon=False)
         plt.tight_layout()
         plt.show()
    else:
        pass
    return loss, val_rmse_0, val_mae_0, test_rmse_0, test_mae_0, best_val_rmse_0, best_epoch_test_rmse_0, best_val_mae_0, best_epoch_test_mae_0, best_val_rmse_1, best_epoch_test_rmse_1, best_val_mae_1, best_epoch_test_mae_1

val_rmse_40_0, test_rmse_40_0  = [], []
val_mae_40_0, test_mae_40_0 = [], []
best_val_rmse_40_0, best_epoch_test_rmse_40_0 = [], []
best_val_mae_40_0, best_epoch_test_mae_40_0 = [], []
best_val_rmse_40_1, best_epoch_test_rmse_40_1 = [], []
best_val_mae_40_1, best_epoch_test_mae_40_1 = [], []


def control_fun():
    for i in range(1, 2):
        out = training(i)
        test_rmse_40_0.append(out[3]) 
        test_mae_40_0.append(out[4])
        val_rmse_40_0.append(out[1])
        val_mae_40_0.append(out[2])
        best_val_rmse_40_0.append(out[5])
        best_epoch_test_rmse_40_0.append(out[6])
        best_val_mae_40_0.append(out[7])
        best_epoch_test_mae_40_0.append(out[8])
        best_val_rmse_40_1.append(out[9])
        best_epoch_test_rmse_40_1.append(out[10])
        best_val_mae_40_1.append(out[11])
        best_epoch_test_mae_40_1.append(out[12])
        print('Seed number ' + str(i))


control_fun()

#Optionally the results are saved in a dataframe. 


df = pd.DataFrame()
df['val_rmse_40_0'] = val_rmse_40_0
df['val_mae_40_0'] =  val_mae_40_0
df['test_rmse_40_0'] = test_rmse_40_0
df['test_mae_40_0'] = test_mae_40_0
df['best_val_rmse_40_0'] = best_val_rmse_40_0
df['best_epoch_test_rmse_40_0'] = best_epoch_test_rmse_40_0
df['best_val_mae_40_0'] = best_val_mae_40_0
df['best_epoch_test_mae_40_0'] = best_epoch_test_mae_40_0
df['best_val_rmse_40_1'] = best_val_rmse_40_1
df['best_epoch_test_rmse_40_1'] = best_epoch_test_rmse_40_1
df['best_val_mae_40_1'] = best_val_mae_40_1
df['best_epoch_test_mae_40_1'] = best_epoch_test_mae_40_1
df['Learning_rate'] = lrate
df['Batch_size'] = batch
df.loc['mean'] = df.mean()
#df.to_excel('.xlsx')
