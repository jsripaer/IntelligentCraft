import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DataFromFile(Dataset):
    def __init__(self, file_path, init_cood=(0,0,0)):
        '''
        file are like 'x.z.npz', containing a region data
        Custom Dataset to load data from multiple numpy files.
        Each file is expected to contain 1024 arrays of shape (24, 4096, 12).
        every element in self.data is a section with shape (4096, 12), 
        in the order of x,z,y
        '''
        self.data = []
        loaded_data = np.load(file_path)
        if len(loaded_data.shape) == 2 and loaded_data.shape[1] == 12:
            self.data.append(loaded_data)
        else:
            raise ValueError(f"Data in {file_path} has incorrect shape: {loaded_data.shape}")
        self.data = np.vstack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        '''
        idx has to be the absolute coordinate of the data point
        then calculate the ralative position in self.data and scanning 8 blocks to each axis
        no more complicated training loop, just take each section in each chunk for training
        '''
        return torch.from_numpy(self.data[indx])
    
class DataLoaderFromFiles:
    def __init__(self, file_paths, batch_size=32, shuffle=True):
        '''
        DataLoader wrapper to create batches from multiple numpy files.
        '''
        dataset = DataFromFile(file_paths)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_loader(self):
        return self.dataloader
    
class TrainFuncs:
    @staticmethod
    def train_step(model, data, target, criterion, optimizer):
        '''On training, in every data , randomly choose few lines
        and fill them into air.
        This take the effects as masks in other models, but there,
        I hope the model can pay attention to every position in the data,
        which is more like the real circumastance.'''
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @staticmethod
    def eval_step(model, data, target, criterion):
        model.eval()
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)
        return loss.item()