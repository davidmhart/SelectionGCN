from argparse import ArgumentParser
import torch
from torch.nn import functional as F
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from torch_geometric.loader import DataLoader

from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal, StaticGraphTemporalSignalBatch

from temporalGNN import TemporalGNN
from my_selection_functions import distance_binning, max_direction, max_dimension, direction_selection, combine_selections, on_road
from torch_geometric_temporal.nn import STConv, MSTGCN, ASTGCN
from my_MSTGCN import SelMSTGCN

import torchmetrics

def batch_dataset(dataset,batch_size,shuffle=False,num_workers=0):
    data_input = np.array(dataset.features)
    data_target = np.array(dataset.targets)

    # Fix for PEMS-BAY
    if data_target.ndim > 3:
        data_target = data_target[:,:,0,:]
    
    data_x_tensor = torch.from_numpy(data_input)
    data_target_tensor = torch.from_numpy(data_target)
    dataset_new = torch.utils.data.TensorDataset(data_x_tensor, data_target_tensor)
    dataloader = torch.utils.data.DataLoader(dataset_new, batch_size=batch_size, shuffle=shuffle,drop_last=True,num_workers=num_workers)
    return dataloader

# Masked function taken from TwoResNet Traffic Prediction Github (https://github.com/semink/TwoResNet)
def masked_metric(agg_fn, error_fn, pred, target, null_value=0.0, agg_dim=0):
    mask = (target != null_value).float()
    target_ = target.clone()
    target_[mask == 0.0] = 1.0  # for mape
    mask /= torch.mean(mask, dim=agg_dim, keepdim=True)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    score = error_fn(pred, target_)
    score = score*mask
    # score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
    return agg_fn(score)


def masked_MAE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mae = masked_metric(agg_fn=lambda e: torch.mean(e, dim=agg_dim),
                        error_fn=lambda p, t: torch.absolute(p - t),
                        pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mae


def masked_MSE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mse = masked_metric(agg_fn=lambda e: torch.mean(e, dim=agg_dim),
                        error_fn=lambda p, t: (p - t) ** 2,
                        pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mse


def masked_RMSE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    rmse = masked_metric(agg_fn=lambda e: torch.sqrt(torch.mean(e, dim=agg_dim)),
                         error_fn=lambda p, t: (p - t)**2,
                         pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return rmse


def masked_MAPE(pred, target, null_value=0.0, agg_dim=(0, 1, 2)):
    mape = masked_metric(agg_fn=lambda e: torch.mean(torch.absolute(e) * 100, dim=agg_dim),
                         error_fn=lambda p, t: ((p - t) / (t)),
                         pred=pred, target=target, null_value=null_value, agg_dim=agg_dim)
    return mape
    
class LitGraphModel(pl.LightningModule):

    def __init__(self, network_name,static_edge_index,in_channels,hidden_size,time_periods,batch_size,static_selections=None,selection_count=0,static_edge_weights=None,mean=0,std=1,num_spatial_layers=1,schedule="epochs30"):
        super().__init__()
        self.register_buffer("edge_index", static_edge_index)
        self.register_buffer("edge_weights", static_edge_weights)
        if static_selections is not None:
            self.register_buffer("selections", static_selections)
            
        self.batch_size = batch_size
        self.network_name = network_name
        self.hidden_size = hidden_size
        self.num_hidden_layers = 1
        if network_name == "STGCN":
            self.recurrent = STConv(int(torch.amax(static_edge_index)+1),in_channels,hidden_size,1,3,3)
        elif network_name == "ASTGCN":
            self.recurrent = ASTGCN(3,in_channels,3,hidden_size,hidden_size,1,time_periods,time_periods,int(torch.amax(static_edge_index)+1))
        elif network_name == "MSTGCN":
            self.recurrent = MSTGCN(3,in_channels,3,hidden_size,hidden_size,1,time_periods,time_periods)
        elif network_name == "SelMSTGCN":
            print("Selection Count:",selection_count)
            self.recurrent = SelMSTGCN(3,in_channels,hidden_size,hidden_size,1,time_periods,time_periods,selection_count,num_spatial_layers)
        else:
            self.recurrent = TemporalGNN(in_channels,hidden_size,time_periods,network_name,selection_count,num_hidden_layers=self.num_hidden_layers,num_spatial_layers=num_spatial_layers)

        self.mean = mean
        self.std = std

        self.schedule = schedule
            
    def configure_optimizers(self):

        if self.schedule == "epochs30":
            lr = 1e-3
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10)
        elif self.schedule == "epochs60":
            lr = 1e-3
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20)
        elif self.schedule == "epochs120":
            lr = 1e-3
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,40)
        # Schedule taken from "Pre-training-Enhanced Spatial-Temporal Graph Neural Network For Multivariate Time Series Forecasting"
        # Does not include gradual curriculum learning
        elif self.schedule == "STEP":
            lr = 5e-3
            optimizer = torch.optim.Adam(self.parameters(), lr=lr,weight_decay=1.0e-5)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1, 18, 36, 54, 72],gamma=0.5)
        else:
            raise ValueError("Training Schedule unknown")

        #self.logger.log_hyperparams(params=dict(network=self.network_name,batch_size=self.batch_size,lr=lr,hidden_size=self.hidden_size,num_hidden_layers=self.num_hidden_layers))

        return [optimizer],[scheduler]

    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        y = train_batch[1]
        
        if self.network_name == "STGCN":
            x = x.permute((0,3,1,2))

        if self.network_name[:3] == "Sel": 
            if self.edge_weights is None:
                h = self.recurrent(x,self.edge_index,self.selections)
            else:
                h = self.recurrent(x,self.edge_index,self.selections,self.edge_weights)    
        else:
            if self.edge_weights is None:
                h = self.recurrent(x,self.edge_index)
            else:
                h = self.recurrent(x,self.edge_index,self.edge_weights)

        if self.network_name == "STGCN":
            h = h.squeeze().permute((0,2,1))
                
        #y = y[:,:,0] # Temp Fix for PEMS-BAY Dataset
            
        h = (h*self.std) + self.mean
        y = (y*self.std) + self.mean
            
        #loss = F.mse_loss(h, y)
        #loss = F.l1_loss(h, y)
        loss = masked_MAE(h,y)
        self.log("train_loss",loss,prog_bar=True,on_step=False,on_epoch=True)
        #self.log("train_mae",mae,prog_bar=True,on_step=False,on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        y = val_batch[1]
        
        if self.network_name == "STGCN":
            x = x.permute((0,3,1,2))
 
        if self.network_name[:3] == "Sel": 
            if self.edge_weights is None:
                h = self.recurrent(x,self.edge_index,self.selections)
            else:
                h = self.recurrent(x,self.edge_index,self.selections,self.edge_weights)    
        else:
            if self.edge_weights is None:
                h = self.recurrent(x,self.edge_index)
            else:
                h = self.recurrent(x,self.edge_index,self.edge_weights)

        if self.network_name == "STGCN":
            h = h.squeeze().permute((0,2,1))
            
        h = (h*self.std) + self.mean
        y = (y*self.std) + self.mean
        
        #import matplotlib.pyplot as plt
        #times = np.arange(5,65,5)
        #plt.plot(times,h[10,100].data.cpu().numpy())
        #plt.plot(times,y[10,100].data.cpu().numpy())
        #plt.show()
        
        #print(self.std)
        #print(self.mean)
        #print(h.shape)
        #print(y.shape)
            
        #loss = torch.sqrt(F.mse_loss(h, y))
        loss = F.l1_loss(h, y)
            
        #self.val_mae(h,y)
        #self.val_rmse(h,y)
        #self.val_mape(h,y)
        
        val_mae = masked_MAE(h, y)
        val_rmse = masked_RMSE(h, y)
        val_mape = masked_MAPE(h, y)
        
        self.log("val_rmse",val_rmse,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("val_mae",val_mae,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("val_mape",val_mape,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x = test_batch[0]
        y = test_batch[1]
        
        if self.network_name == "STGCN":
            x = x.permute((0,3,1,2))

        if self.network_name[:3] == "Sel": 
            if self.edge_weights is None:
                h = self.recurrent(x,self.edge_index,self.selections)
            else:
                h = self.recurrent(x,self.edge_index,self.selections,self.edge_weights)    
        else:
            if self.edge_weights is None:
                h = self.recurrent(x,self.edge_index)
            else:
                h = self.recurrent(x,self.edge_index,self.edge_weights)

        if self.network_name == "STGCN":
            h = h.squeeze().permute((0,2,1))
            
        h = (h*self.std) + self.mean
        y = (y*self.std) + self.mean
            
        #loss = torch.sqrt(F.mse_loss(h, y))
        loss = F.l1_loss(h, y)
        
        #self.test_rmse(h,y)
        #self.test_mae(h,y)
        #self.test_mape(h,y)

        test_mae = masked_MAE(h, y)
        test_rmse = masked_RMSE(h, y)
        test_mape = masked_MAPE(h, y)
        
        self.log("test_rmse_overall",test_rmse,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_mae_overall",test_mae,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_mape_overall",test_mape,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        #self.test_rmse15(h[:,:,2],y[:,:,2])
        #self.test_mae15(h[:,:,2],y[:,:,2])
        #self.test_mape15(h[:,:,2],y[:,:,2])
        
        test_rmse15 = masked_RMSE(h[:,:,2],y[:,:,2],agg_dim=(0, 1))
        test_mae15 = masked_MAE(h[:,:,2],y[:,:,2],agg_dim=(0, 1))
        test_mape15 = masked_MAPE(h[:,:,2],y[:,:,2],agg_dim=(0, 1))
        
        self.log("test_rmse_15min", test_rmse15,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_mae_15min",test_mae15,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_mape_15min",test_mape15,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        #self.test_rmse30(h[:,:,5],y[:,:,5])
        #self.test_mae30(h[:,:,5],y[:,:,5])
        #self.test_mape30(h[:,:,5],y[:,:,5])
        
        test_rmse30 = masked_RMSE(h[:,:,5],y[:,:,5],agg_dim=(0, 1))
        test_mae30 = masked_MAE(h[:,:,5],y[:,:,5],agg_dim=(0, 1))
        test_mape30 = masked_MAPE(h[:,:,5],y[:,:,5],agg_dim=(0, 1))
        
        self.log("test_rmse_30min",test_rmse30,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_mae_30min",test_mae30,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_mape_30min",test_mape30,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        #self.test_rmse60(h[:,:,11],y[:,:,11])
        #self.test_mae60(h[:,:,11],y[:,:,11])
        #self.test_mape60(h[:,:,11],y[:,:,11])
        
        test_rmse60 = masked_RMSE(h[:,:,11],y[:,:,11],agg_dim=(0, 1))
        test_mae60 = masked_MAE(h[:,:,11],y[:,:,11],agg_dim=(0, 1))
        test_mape60 = masked_MAPE(h[:,:,11],y[:,:,11],agg_dim=(0, 1))
        
        self.log("test_rmse_60min",test_rmse60,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)
        self.log("test_mae_60min",test_mae60,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True) 
        self.log("test_mape_60min",test_mape60,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        return loss


def train_network(dataset_name,network_name,selection_function,selection_count,batch_size,hidden_size,num_spatial,schedule,devices=[0],num_workers=0,use_edge_weights=False):
    
    if schedule == "epochs30":
        max_epochs = 30
    elif schedule == "epochs60":
        max_epochs = 60
    elif schedule == "epochs120":
        max_epochs = 120
    elif schedule == "STEP":
        max_epochs = 100
    else:
        raise ValueError("Training Schedule unknown")


    if dataset_name == "METRLA":
        loader = METRLADatasetLoader()
        in_channels=2
        time_periods=12
        
        if network_name == "STGCN":
            dataset = loader.get_dataset(num_timesteps_in=16, num_timesteps_out=time_periods)
        else:
            dataset = loader.get_dataset(num_timesteps_in=time_periods, num_timesteps_out=time_periods)
            
        train_loader, test_loader = temporal_signal_split(dataset, train_ratio=0.8)
        train_loader, val_loader = temporal_signal_split(train_loader,train_ratio=0.9)
        
        means=[53.59967,0.4982691]
        stds=[20.209862,0.28815305]
        
        import pandas as pd
        #df = pd.read_csv("METRLA_graph_sensor_locations.csv")
        df = pd.read_csv("sensor_locations_METRLA.csv")
        latitude = df["latitude"]
        longitude = df["longitude"]
        location = np.stack((latitude,longitude),axis=1)
        
        location = torch.tensor(location,dtype=torch.float)


    elif dataset_name == "PEMS-BAY":
        loader = PemsBayDatasetLoader()
        in_channels = 2
        time_periods = 12
        dataset = loader.get_dataset(num_timesteps_in=time_periods, num_timesteps_out=time_periods)
        train_loader, test_loader = temporal_signal_split(dataset, train_ratio=0.8)
        train_loader, val_loader = temporal_signal_split(train_loader,train_ratio=0.9)
        
        means=[61.77375,0.4984733]
        stds=[9.293026,0.28541598]
        
        import pandas as pd
        df = pd.read_csv("sensor_locations_PEMS-BAY.csv")
        latitude = df["Latitude"]
        longitude = df["Longitude"]
        location = np.stack((latitude,longitude),axis=1)
        
        location = torch.tensor(location,dtype=torch.float)
        
    else:
        raise ValueError("dataset_name unknown")
        
    static_edge_index = train_loader[0].edge_index
    static_edge_weights = train_loader[0].edge_attr
    

    if network_name[:3] == "Sel":
        if selection_function == "distance":
            static_selections = distance_binning(static_edge_weights,selection_count)
        elif selection_function == "distance_log":
            static_selections = distance_binning(static_edge_weights,selection_count,log=True)
        elif selection_function == "direction":
            sel_f = direction_selection(selection_count-1)
            static_selections = sel_f(location,static_edge_index)
        elif selection_function == "distance_direction":
            dist_selections = distance_binning(static_edge_weights,selection_count)
            sel_f = direction_selection(4)
            dir_selections = sel_f(location,static_edge_index)
            static_selections = combine_selections(dist_selections,dir_selections)
            selection_count = 4 * (selection_count - 1) + 1
        elif selection_function == "on_road":
            static_selections = on_road(location,static_edge_index)
            selection_count = 2
        else:
            raise ValueError("Selection Function unknown")
    else:
        selection_function = "baseline" # For logging purposes
        selection_count = 0
        static_selections = None

    use_edge_weights = bool(use_edge_weights)
        
    if not use_edge_weights:
        static_edge_weights = None

    # Batch data
    train_loader = batch_dataset(train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = batch_dataset(val_loader, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = batch_dataset(test_loader, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    
    if use_edge_weights:
        log_directory = "logs/" + dataset_name + "/" + network_name + str(num_spatial) + "(EW)/" + selection_function + "(" + str(selection_count) + ")/"
    else:
        log_directory = "logs/" + dataset_name + "/" + network_name + str(num_spatial)+ "/" + selection_function + "(" + str(selection_count) + ")/"
    logger = pl_loggers.TensorBoardLogger(save_dir=log_directory)
    
    logger.log_hyperparams(params=dict(network=network_name,batch_size=batch_size,lr=1e-3,hidden_size=hidden_size,num_hidden_layers=1,num_spatial_layers=num_spatial,selection_function=selection_function,selection_count=selection_count))

    #print(static_edge_weights)
    model = LitGraphModel(network_name,static_edge_index,in_channels,hidden_size,time_periods,batch_size,static_selections,selection_count,static_edge_weights,means[0],stds[0],num_spatial_layers=num_spatial,schedule=schedule)

    #early_stop_callback = EarlyStopping(monitor='val_mse',min_delta=0.00,patience=10,verbose=False,mode='max')
    #trainer = pl.Trainer(logger=logger,accelerator="gpu", devices=devices,callbacks=[checkpoints,early_stop_callback])

    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_rmse', save_top_k=1, mode='min')
    #checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_mae', save_top_k=1, mode='min')

    #trainer = pl.Trainer(logger=logger,max_epochs=max_epochs,accelerator="cpu",callbacks=[checkpoint])
    trainer = pl.Trainer(logger=logger,max_epochs=max_epochs,accelerator="gpu",devices=devices,callbacks=[checkpoint])
    trainer.fit(model, train_loader, val_loader) 
    trainer.test(ckpt_path="best", dataloaders=test_loader)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_name", type=str, default="METRLA") 
    parser.add_argument("network_name", type=str, default="GCN") # Give the name of the spatial aggregator you want inside TemporalGNN
    parser.add_argument("--selection_function", type=str, default="distance")
    parser.add_argument("--selection_count", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_spatial", type=int, default=1)
    parser.add_argument("--schedule", type=str, default="epochs30")
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_edge_weights", type=int, default=0)
    args = parser.parse_args()
    
    train_network(
        args.dataset_name,
        args.network_name,
        args.selection_function,
        args.selection_count,
        args.batch_size,
        args.hidden_size,
        args.num_spatial,
        args.schedule,
        args.devices,
        args.num_workers,
        args.use_edge_weights
    )
