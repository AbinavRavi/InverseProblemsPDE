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
        return int(len(self.data)/5)
    
    def getParameters(self, path):
        newPath=path.split('/')[:-1]
        newPath='/'.join(newPath)
        parameterfile = newPath+'_parameters.csv'
        return parameterfile

    def __getitem__(self,index):
        res = (index+1)%50
        index = np.random.randint(0, 10)+res*50
#         if (index+1)%50==0:
#             index-=1
        parameterfile = self.getParameters(self.data[index])
        tmp = []
        for i in range(10):
            tmp.append(np.expand_dims(self.csvreader(self.data[index+i]),axis=0))
        return np.squeeze(self.csvreader(parameterfile)),np.expand_dims(self.csvreader(self.data[index]),axis=0), np.array(tmp)
    
        
