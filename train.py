from __future__ import print_function, division

import torch
# from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import numpy as np
# import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import pandas as pd



class AGIDataset(Dataset):
	"""AGI Dataset, expects two AGI vectors """
	def __init__(self, data_file, train_mode=True, transform=None):
		self.data_file = data_file
		self.transform = transform

		self.raw_data = pd.read_table(data_file, delim_whitespace=True, header=0)
		self.raw_data.AGIVec  = self.raw_data.AGIVec.apply(lambda x : [float(y) for y in x.split(',')])
		self.raw_data.AGIVecQ = self.raw_data.AGIVecQ.apply(lambda x : [float(y) for y in x.split(',')])
		self.raw_data = self.raw_data[['AGIVec']]
		self.train_mode = train_mode


		if self.train_mode:
			# self.data = [self.raw_data.iloc[i][0:5].tolist()  for i in range(self.raw_data.shape[0])]
			self.data = self.raw_data.as_matrix()
		else:
			self.data = [self.raw_data.iloc[i][0:3].tolist()  for i in range(self.raw_data.shape[0])]		


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return (self.data[idx])



data = AGIDataset("data.tsv")
print(data)
print(type(data))

print(data[1])
# p = torch.utils.data.DataLoader(data, batch_size=3, shuffle=False)
# count = 0
# for i,x in enumerate(p):
# 	count = count + 1
	
	# d = Variable(torch.Tensor(x[2]))
	# print(x[])
	# print(type(x[2][0]))
