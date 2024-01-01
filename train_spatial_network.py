from argparse import ArgumentParser

import os.path as osp

import torch

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import my_datasets
import my_networks
from my_selection_functions import direction_selection, direction3D_selection, distance_selection, direction_distance_selection

class LitGraph(pl.LightningModule):
    def __init__(self,network_name,n_features,n_classes,features_name,selection_criteria,selection_function,selection_count,lr=1e-3,hidden_size=256,num_layers=3,batch_size=1):
        super().__init__()

        if network_name == "GCN":
            self.network = my_networks.GCN(n_features,n_classes,hidden_size,num_layers=num_layers)
        elif network_name == "SelGCN":
            self.network = my_networks.SelGCN(n_features,n_classes,hidden_size,selection_function=selection_function,selection_count=selection_count,num_layers=num_layers)
        elif network_name == "SAGEConv":
            self.network = my_networks.SAGEConvNet(n_features,n_classes,hidden_size,num_layers=num_layers)
        elif network_name == "SelSAGEConv":
            self.network = my_networks.SelSAGEConvNet(n_features,n_classes,hidden_size,selection_function=selection_function,selection_count=selection_count,num_layers=num_layers)
        elif network_name == "GraphConv":
            self.network = my_networks.GraphConvNet(n_features,n_classes,hidden_size,num_layers=num_layers)
        elif network_name == "SelGraphConv":
            self.network = my_networks.SelGraphConvNet(n_features,n_classes,hidden_size,selection_function=selection_function,selection_count=selection_count,num_layers=num_layers)
        elif network_name == "GINConv":
            self.network = my_networks.GINConvNet(n_features,n_classes,hidden_size,num_layers=num_layers)
        elif network_name == "SelGINConv":
            self.network = my_networks.SelGINConvNet(n_features,n_classes,hidden_size,selection_function=selection_function,selection_count=selection_count,num_layers=num_layers)
        elif network_name == "GAT":
            self.network = my_networks.GAT(n_features,n_classes,hidden_size,num_layers=num_layers)
        elif network_name == "TransformerConv":
            self.network = my_networks.TransformerConvNet(n_features,n_classes,hidden_size,num_layers=num_layers)
        elif network_name == "EdgeConv":
            self.network = my_networks.EdgeConvNet(n_features,n_classes,hidden_size,num_layers=num_layers)
        elif network_name == "DynamicEdgeConv":
            self.network = my_networks.DynamicEdgeConvNet(n_features,n_classes,hidden_size,num_layers=num_layers)
        else:
            raise ValueError("network_name unknown")

        self.network_name = network_name
        self.features_name = features_name
        self.selection_criteria = selection_criteria
        self.learning_rate = lr
        self.batch_size = batch_size

        self.global_pool = True
        self.criteria = torch.nn.CrossEntropyLoss()

        self.train_F1 = torchmetrics.F1Score("multiclass",num_classes=n_classes,average='micro')
        self.val_F1 = torchmetrics.F1Score("multiclass",num_classes=n_classes,average='micro')
        self.train_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.val_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')
        self.test_F1 = torchmetrics.F1Score("multiclass",num_classes=n_classes,average='micro')
        self.test_acc = torchmetrics.Accuracy("multiclass",num_classes=n_classes,average='micro')

    def training_step(self, data, batch_idx):

        if self.features_name == "x+pos":
            features = torch.cat((data.x,data.pos),dim=1)
        elif self.features_name == "ones":
            features = torch.ones((torch.amax(data.edge_index),1),dtype=torch.float,device=data.edge_index.device)
        else:
            features = getattr(data,self.features_name)

        if self.network_name[:3] == "Sel":
            sel_criteria = getattr(data,self.selection_criteria)
            outs = self.network(features,sel_criteria,data.edge_index)
        elif self.network_name[:3] == "Dyn":
            outs = self.network(features,data.edge_index,data.batch)
        else:
            outs = self.network(features,data.edge_index)

        if self.global_pool:
            outs = my_networks.max_pool(outs,data.batch)

        loss = self.criteria(outs, data.y)
        self.log("train_loss",loss,batch_size=self.batch_size,sync_dist=True)

        self.train_F1(outs,data.y.long())
        self.log("train_F1",self.train_F1,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        self.train_acc(outs,data.y.long())
        self.log("train_acc",self.train_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20)
        return [optimizer],[scheduler]

    def validation_step(self, val_data, batch_idx):
        
        if self.features_name == "x+pos":
            features = torch.cat((val_data.x,val_data.pos),dim=1)
        elif self.features_name == "ones":
            features = torch.ones((torch.amax(val_data.edge_index),1),dtype=torch.float,device=val_data.edge_index.device)
        else:
            features = getattr(val_data,self.features_name)

        if self.network_name[:3] == "Sel":
            sel_criteria = getattr(val_data,self.selection_criteria)
            outs = self.network(features,sel_criteria,val_data.edge_index)
        elif self.network_name[:3] == "Dyn":
            outs = self.network(features,val_data.edge_index,val_data.batch)
        else:
            outs = self.network(features,val_data.edge_index)


        if self.global_pool:
            outs = my_networks.max_pool(outs,val_data.batch)

        loss = self.criteria(outs, val_data.y)
        self.log("val_loss",loss,batch_size=self.batch_size,sync_dist=True)

        self.val_F1(outs,val_data.y.long())
        self.log("val_F1",self.val_F1,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        self.val_acc(outs,val_data.y.long())
        self.log("val_acc",self.val_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        return loss

    def test_step(self, test_data, batch_idx):

        if self.features_name == "x+pos":
            features = torch.cat((test_data.x,test_data.pos),dim=1)
        elif self.features_name == "ones":
            features = torch.ones((torch.amax(test_data.edge_index),1),dtype=torch.float,device=test_data.edge_index.device)
        else:
            features = getattr(test_data,self.features_name)

        if self.network_name[:3] == "Sel":
            sel_criteria = getattr(test_data,self.selection_criteria)
            outs = self.network(features,sel_criteria,test_data.edge_index)
        elif self.network_name[:3] == "Dyn":
            outs = self.network(features,test_data.edge_index,test_data.batch)
        else:
            outs = self.network(features,test_data.edge_index)


        if self.global_pool:
            outs = my_networks.max_pool(outs,test_data.batch)
        
        loss = self.criteria(outs, test_data.y)

        self.test_F1(outs,test_data.y.long())
        self.log("test_F1",self.test_F1,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        self.test_acc(outs,test_data.y.long())
        self.log("test_acc",self.test_acc,prog_bar=True,on_step=False,on_epoch=True,sync_dist=True)

        return loss



def train_network(dataset_name,network_name,features_name,selection_criteria,selection_function_name,selection_count,batch_size,hidden_size,num_layers,learning_rate,max_epochs,devices=[0],num_workers=0):

    if dataset_name == "MNIST":
        dataset = my_datasets.MNISTsup_dataset(batch_size=batch_size,num_workers=num_workers)
    elif dataset_name == "CoMA":
        dataset = my_datasets.CoMA_dataset(batch_size=batch_size,num_workers=num_workers)
    else:
        raise ValueError("dataset_name unknown")

    if features_name == "x+pos":
        num_features = 3
    elif features_name == "ones":
        num_features = 1
    else:
        num_features = dataset.num_features
    
    if network_name[:3] == "Sel":
        if selection_function_name == "direction":
            selection_function = direction_selection(selection_count-1)
        elif selection_function_name == "direction3D":
            selection_function = direction3D_selection(selection_count-1)
        elif selection_function_name == "distance":
            selection_function = distance_selection(selection_count)
        elif selection_function_name == "direction_distance":
            selection_function,selection_count = direction_distance_selection(selection_count-1,4)
        elif selection_function_name == "direction3D_distance":
            selection_function,selection_count = direction_distance_selection(selection_count-1,4,use3D=True)
        else:
            raise ValueError("Selection Function unknown")
    else:
        selection_function_name = "-" # For logging purposes
        selection_function = None
        selection_count = 0
    

    if network_name[:3] == "Sel":
        log_directory = "logs/"+dataset_name+"/"+network_name+"("+features_name+","+selection_function_name+","+"{:02}".format(selection_count)+")"
    else:
        log_directory = "logs/"+dataset_name+"/"+network_name+"("+features_name+")"

    model = LitGraph(network_name,num_features,dataset.num_classes,features_name,selection_criteria,selection_function,selection_count,learning_rate,hidden_size=hidden_size,num_layers=num_layers,batch_size=batch_size)
    logger = pl_loggers.TensorBoardLogger(save_dir=log_directory)
    
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_F1', save_top_k=1, mode='max')

    #trainer = pl.Trainer(logger=logger,max_epochs=max_epochs, accelerator="cpu", callbacks=[checkpoint])
    trainer = pl.Trainer(logger=logger,max_epochs=max_epochs, accelerator="gpu", devices=devices,callbacks=[checkpoint])
    trainer.fit(model,dataset.train_loader,dataset.val_loader)
    
    trainer.test(ckpt_path="best", dataloaders=dataset.test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset_name", type=str, default="MNIST") 
    parser.add_argument("network_name", type=str, default="GCN") 
    parser.add_argument("features_name", type=str, default="x") #["x","pos","x+pos","ones"]
    parser.add_argument("--selection_criteria", type=str, default="pos") #["x","pos"]
    parser.add_argument("--selection_function", type=str, default="direction")
    parser.add_argument("--selection_count", type=int, default = 9)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    train_network(
        args.dataset_name,
        args.network_name,
        args.features_name,
        args.selection_criteria,
        args.selection_function,
        args.selection_count,
        args.batch_size,
        args.hidden_size,
        args.num_layers,
        args.learning_rate,
        args.max_epochs,
        args.devices,
        args.num_workers,
    )
    
