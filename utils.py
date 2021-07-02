import torch
import os
import imageio
import matplotlib.pyplot as plt


def plot_loss_curve(train_list):
    plt.figure()
    plt.title('Gaussian_Process')
    plt.plot(train_list, 'r-', label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('NLL')
    plt.legend(loc='upper right')
    plt.savefig('plots/train_loss.png')
    plt.close()

    
def plot_regression(x, y, long_x, long_y, xs, y_dist, ktype1, ktype2='', otype='', idx=''):       
    x, y, xs = torch.squeeze(x).cpu(), torch.squeeze(y).cpu(), torch.squeeze(xs).cpu()
    long_x, long_y = torch.squeeze(long_x).cpu(), torch.squeeze(long_y).cpu()
    
    mean, std = y_dist.mean.detach(), y_dist.stddev.detach()
    mean, std = torch.squeeze(mean).cpu(), torch.squeeze(std).cpu()
    
    plt.figure()
    plt.title('regression')
    plt.plot(long_x, long_y, 'g.', markersize=10)
    plt.plot(x, y, 'r.', markersize=10, label='data')
    plt.plot(xs, mean, 'b-', label='Predictions')
    plt.fill(torch.cat((xs, torch.flip(xs, [0])),0),
             torch.cat((mean - 1.9600 * std, torch.flip(mean + 1.9600 * std, [0])),0),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper right')
    plt.savefig('plots/regression_'+ktype1+ktype2+otype+str(idx)+'.png')
    plt.close()
