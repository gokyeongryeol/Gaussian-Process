import os
import math
import pandas as pd
import numpy as np
from utils import *
from GP_model import * 
import matplotlib

matplotlib.use('Agg')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('data/korea_data_part.csv', header=None)
long_data = pd.read_csv('data/korea_data_add.csv', header=None)
test = pd.read_csv('data/test.csv', header=None)

x = torch.Tensor(np.asarray(data[0])).view(-1, 1).to(device)
y = torch.Tensor(np.asarray(data[1])).view(-1, 1).to(device)
long_x = torch.Tensor(np.asarray(long_data[0])).view(-1, 1).to(device)
long_y = torch.Tensor(np.asarray(long_data[1])).view(-1, 1).to(device)
xs = torch.Tensor(np.asarray(test[0])).view(-1, 1).to(device)

n_epoch = 100000


def main(ktype):
    model = GaussianProcess(ktype).to(device)
    model.load_state_dict(torch.load('models/GP_'+ktype+'.pt'))
    optimizer = optim.Adam(model.parameters(), 1)
    
    train_list = []
    for epoch in range(n_epoch):
        loss = model(x, y)
        train_list.append(loss.item())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (epoch+1) % 1 == 0:
            print(loss.item())
            y_dist = model.prediction(x, y, xs)
            plot_regression(x, y, long_x, long_y, xs, y_dist, ktype)
            plot_loss_curve(train_list)
        torch.save(model.state_dict(), 'models/GP_'+ktype+'.pt')
            

if __name__ == '__main__': 
    main('LIN')