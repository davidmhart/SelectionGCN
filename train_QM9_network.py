from argparse import ArgumentParser

import os.path as osp

import torch

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
import my_datasets
import my_networks
from torch.nn import functional as F
from torch_geometric.utils import scatter

from torch_geometric.nn import DimeNet, DimeNetPlusPlus
from my_dimenet import SelDimeNet, SelDimeNetPlusPlus, SelDimeNetPPmod
from torch_geometric.datasets import QM9

from my_selection_functions import direction3D_selection, distance_selection, attribute_selection


class LitGraph(pl.LightningModule):
    def __init__(self,target_num,network_name,n_features,n_classes,features_name,selection_criteria=None,selection_function=None,selection_count=None,lr=1e-3,schedule_type="epochs100",use_ema=False,hidden_size=256,batch_size=1):
        super().__init__()

        if network_name == "GCN":
            self.network = my_networks.GCN(n_features,n_classes,hidden_size)
        elif network_name == "DimeNet":
            self.network = DimeNet(128,1,6,8,7,6) # Numbers taken from PyG DimeNet._from_QM9 function
        elif network_name == "DimeNetsmall":
            self.network = DimeNet(64,1,6,8,7,6)
        elif network_name == "DimeNetPP":
            self.network = DimeNetPlusPlus(128,1,4,64,8,256,7,6) # Numbers taken from PyG DimeNet._from_QM9 function
        elif network_name == "DimeNetPPsmall":
            self.network = DimeNetPlusPlus(64,1,4,64,8,64,7,6) 
        elif network_name == "SelDimeNet":
            self.network = SelDimeNet(128,1,6,8,7,6,selection_count_dist=selection_count[0],selection_count_angle=selection_count[1]) # Numbers taken from PyG example script
        elif network_name == "SelDimeNetsmall":
            self.network = SelDimeNet(64,1,6,8,7,6,selection_count_dist=selection_count[0],selection_count_angle=selection_count[1])
        elif network_name == "SelDimeNetPP":
            self.network = SelDimeNetPlusPlus(128,1,4,64,8,256,7,6,selection_count_dist=selection_count[0],selection_count_angle=selection_count[1]) # Numbers taken from PyG example script
        elif network_name == "SelDimeNetPPsmall":
            self.network = SelDimeNetPlusPlus(64,1,4,64,8,64,7,6,selection_count_dist=selection_count[0],selection_count_angle=selection_count[1]) # Numbers taken from PyG example script
        elif network_name == "SelDimeNetPPmod":
            self.network = SelDimeNetPPmod(128,1,4,64,8,256,7,6,selection_count_dist=selection_count[0],selection_count_angle=selection_count[1]) # Numbers taken from PyG example script
        else:
            raise ValueError("network_name unknown")

        self.network_name = network_name
        self.features_name = features_name
        self.selection_criteria = selection_criteria
        self.learning_rate = lr
        self.schedule_type = schedule_type
        self.use_ema = use_ema
        self.batch_size = batch_size
        
        self.train_loss = "train_loss"+"{:02}".format(target_num)
        self.val_mse = "val_mse"+"{:02}".format(target_num)
        self.val_mae = "val_mae"+"{:02}".format(target_num)
        self.test_mse = "test_mse"+"{:02}".format(target_num)
        self.test_mae = "test_mae"+"{:02}".format(target_num)
        
        self.target_num = target_num
        
        if self.use_ema:
            # Include Exponential Moving Average (from Pytorch Docs)
            ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: 0.001 * averaged_model_parameter + 0.999 * model_parameter
            self.ema_model = torch.optim.swa_utils.AveragedModel(self.network,avg_fn=ema_avg)
        
    def forward(self,data):
        
        if self.features_name == "z+pos":
            features = torch.cat((data.z.unsqueeze(1),data.pos),dim=1)
        elif self.features_name == "x+pos":
            features = torch.cat((data.x,data.pos),dim=1)
        else:
            features = getattr(data,self.features_name)

        if "DimeNet" in self.network_name:
            outs = self.network(data.z, data.pos, data.batch).squeeze()
            # Since the network pre scatters the output, return directly
            return outs 
        elif self.network_name[:3] == "Sel":
            sel_criteria = getattr(data,self.selection_criteria)
            outs = self.network(features,sel_criteria,data.edge_index)
        elif self.network_name[:3] == "Dyn":
            outs = self.network(features,data.edge_index,data.batch)
        else:
            outs = self.network(features,data.edge_index)

        result = scatter(outs, data.batch, dim=0, reduce='sum').squeeze()
        
        return result
    
    def forward_ema(self,data):
        
        if self.features_name == "z+pos":
            features = torch.cat((data.z.unsqueeze(1),data.pos),dim=1)
        elif self.features_name == "x+pos":
            features = torch.cat((data.x,data.pos),dim=1)
        else:
            features = getattr(data,self.features_name)

        if "DimeNet" in self.network_name:
            outs = self.ema_model(data.z, data.pos, data.batch).squeeze()
            # Since the network pre scatters the output, return directly
            return outs 
        elif self.network_name[:3] == "Sel":
            sel_criteria = getattr(data,self.selection_criteria)
            outs = self.ema_model(features,sel_criteria,data.edge_index)
        elif self.network_name[:3] == "Dyn":
            outs = self.ema_model(features,data.edge_index,data.batch)
        else:
            outs = self.ema_model(features,data.edge_index)

        result = scatter(outs, data.batch, dim=0, reduce='sum').squeeze()
        
        return result
    
    def training_step(self, data, batch_idx):

        result = self.forward(data)
        
        #loss = F.mse_loss(result, data.y)
        loss = F.l1_loss(result, data.y)
        self.log(self.train_loss,loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        if self.schedule_type == "epochs100":
            optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,25)
        
        elif self.schedule_type == "dimenet_short":
            # Approximate implementation of LinearWarmupExpontialDecay from DimeNet tensorflow implementation
            optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1/3000,end_factor=1,total_iters=3000)
            scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.01**(1/200000))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,[scheduler1,scheduler2],[3000])
            lr_scheduler = {"scheduler": scheduler, "interval":"step"}
        
        elif self.schedule_type == "dimenet":
            # Approximate implementation of LinearWarmupExpontialDecay from DimeNet tensorflow implementation
            optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1/3000,end_factor=1,total_iters=3000)
            scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.01**(1/4000000))
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,[scheduler1,scheduler2],[3000])
            lr_scheduler = {"scheduler": scheduler, "interval":"step"}
        
        else:
            raise ValueError("schedule_type unknown")
            
        return [optimizer],[lr_scheduler]

    def on_before_zero_grad(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.update_parameters(self.network)

    def validation_step(self, val_data, batch_idx):
        
        if self.use_ema:
            result = self.forward_ema(val_data)
        else:
            result = self.forward(val_data)
            
        loss = F.mse_loss(result, val_data.y)
        mae = F.l1_loss(result, val_data.y)
        
        # Report meV instead of eV.
        mae = 1000 * mae if self.target_num in [2, 3, 4, 6, 7, 8, 9, 10] else mae
        
        self.log(self.val_mse,loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log(self.val_mae,mae,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.batch_size,sync_dist=True)

        return loss

    def test_step(self, test_data, batch_idx):

        if self.use_ema:
            result = self.forward_ema(test_data)
        else:
            result = self.forward(test_data)
                
        loss = F.mse_loss(result, test_data.y)
        mae = F.l1_loss(result, test_data.y)
        
        # Report meV instead of eV.
        mae = 1000 * mae if self.target_num in [2, 3, 4, 6, 7, 8, 9, 10] else mae
        
        self.log(self.test_mse,loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.batch_size,sync_dist=True)
        self.log(self.test_mae,mae,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.batch_size,sync_dist=True)        
        
        return loss



def train_network(target_num,network_name,features_name,selection_function_name,selection_count,batch_size,hidden_size,learning_rate,schedule_type,use_ema,devices=[0],num_workers=0):

    # Convert to bool
    use_ema = use_ema > 0
    
    dataset = my_datasets.QM9_dataset(target_num=target_num,batch_size=batch_size,num_workers=num_workers)
        
    if features_name == "z+pos":
        num_features = 4
    elif features_name == "z":
        num_features = 1
    elif features_name == "pos":
        num_features = 3
    elif features_name == "x":
        num_features = 11
    elif features_name == "x+pos":
        num_features = 14
    else:
        num_features = dataset.num_features
        
    if selection_function_name == "Direction":
        selection_criteria = "pos"
        selection_count = 27
        selection_function = direction3D_selection(selection_count-1)
    elif selection_function_name == "BondType":
        selection_criteria = "edge_attr"
        selection_count = 4
        selection_function = attribute_selection(0,4)
    elif selection_function_name == "AtomType":
        selection_criteria = "x"
        selection_count = 5
        selection_function = attribute_selection(0,5)
    elif selection_function_name == "AtomRelation":
        selection_criteria = "x"
        selection_count = 25
        selection_function = attribute_selection(0,5,True)
    elif selection_function_name == "distance_angle":
        # Specific to SelDimeNet
        selection_function = None
        selection_criteria = None
        # Distance Selections, Angle Selections
        #selection_count = [10,0]
        if selection_count[0] == 0:
            selection_function_name = "angle"+str(selection_count[1])
        elif selection_count[1] == 0:
            selection_function_name = "dist"+str(selection_count[0])
        else:
            selection_function_name = "dist"+str(selection_count[0]) + "_angle"+str(selection_count[1])
    else:
        selection_function_name = "-"
        selection_function = None
        selection_criteria = None
        selection_count = None
    
    
    if network_name[:3] == "Sel":
        log_directory = "logs/QM9/"+"{:02}".format(target_num)+"/"+selection_function_name+"/"+network_name+"("+features_name+",schedule="+schedule_type+",EMA="+str(use_ema)+")"
    else:
        log_directory = "logs/QM9/"+"{:02}".format(target_num)+"/baseline/"+network_name+"("+features_name+",schedule="+schedule_type+",EMA="+str(use_ema)+")"

    model = LitGraph(target_num,network_name,num_features,dataset.num_classes,features_name,selection_criteria,selection_function,selection_count,learning_rate,schedule_type,use_ema,hidden_size=hidden_size,batch_size=batch_size)
    logger = pl_loggers.TensorBoardLogger(save_dir=log_directory)
    
    #checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_mse'+"{:02}".format(target_num), save_top_k=1, mode='min')
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_mae'+"{:02}".format(target_num), save_top_k=1, mode='min')
    
    if schedule_type == "epochs100":     
        #trainer = pl.Trainer(logger=logger,max_epochs=max_epochs, accelerator="cpu", callbacks=[checkpoint])
        trainer = pl.Trainer(logger=logger,max_epochs=100, accelerator="gpu", devices=devices,callbacks=[checkpoint])
    elif schedule_type == "dimenet_short":
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(logger=logger,max_steps=170000, accelerator="gpu", devices=devices,callbacks=[checkpoint,lr_monitor])
    elif schedule_type == "dimenet":
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer = pl.Trainer(logger=logger,max_steps=3000000, accelerator="gpu", devices=devices,callbacks=[checkpoint,lr_monitor])
    
    trainer.fit(model,dataset.train_loader,dataset.val_loader)
    
    trainer.test(ckpt_path="best", dataloaders=dataset.test_loader)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("target_num", type=int) 
    parser.add_argument("network_name", type=str, default="GCN")
    parser.add_argument("--features_name", type=str, default="z+pos")
    parser.add_argument("--selection_function", type=str, default="baseline")
    parser.add_argument("--selection_count", nargs="+", type=int, default=[1])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--schedule_type", type=str, default="dimenet_short")
    parser.add_argument("--use_ema", type=int, default=1)
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    train_network(
        args.target_num,
        args.network_name,
        args.features_name,
        args.selection_function,
        args.selection_count,
        args.batch_size,
        args.hidden_size,
        args.learning_rate,
        args.schedule_type,
        args.use_ema,
        args.devices,
        args.num_workers,
    )
    
