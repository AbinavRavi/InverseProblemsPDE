import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import time
import glob
import csv
import pandas as pd

class Ipde(Dataset):
    def __init__(self, path):
        self.data = glob.glob(path+'/**/*.csv')
        self.data = sorted(self.data, key=lambda a: int(os.path.split(os.path.split(a)[0])[1].zfill(3)+os.path.split(a)[1][:-4].zfill(3)))
        #print(len(self.data))
        
    def csvreader(self,filename):
        data = pd.read_csv(filename,header=None)
        return np.array(data)
        
    def __len__(self):
        return len(self.data)
    
    def getParameters(self,path):
        newPath=path.split('/')[:-1]
        newPath='/'.join(newPath)
        parameterfile = newPath+'_parameters.csv'
        return parameterfile

    def __getitem__(self,index):
        if (index+1)%50==0:
            index-=1
        parameterfile = self.getParameters(self.data[index])
        return np.squeeze(self.csvreader(parameterfile)),np.expand_dims(self.csvreader(self.data[index]),axis=0),np.expand_dims(self.csvreader(self.data[index+1]),axis=0)
    
        
