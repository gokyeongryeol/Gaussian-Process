import math
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class GaussianProcess(nn.Module):
    def __init__(self, ktype):
        super(GaussianProcess, self).__init__()
        self.ktype = ktype
        
        self.sigma0 = nn.Parameter(torch.Tensor([1]))
        
        self.constant0 = nn.Parameter(torch.Tensor([1]))  
        
        self.constant1 = nn.Parameter(torch.Tensor([1]))  
        self.coeff1 = nn.Parameter(torch.Tensor([1]))
        
        self.l1 = nn.Parameter(torch.Tensor([1]))
        self.sigma1 = nn.Parameter(torch.Tensor([1]))
        
        self.error = torch.Tensor([1])
        

    def kernel(self, m, n, ktype):
        if ktype == 'WN':
            m = torch.unsqueeze(m, 1)
            n = torch.unsqueeze(n, 0)
            norm = torch.norm(m-n, dim=-1)
            kernel = self.sigma0**2 * (norm == 0)
            return kernel
        
        elif ktype == 'LIN':
            kernel = self.constant1**2 + self.coeff1 * torch.ger(m.view(-1)-self.constant0, n.view(-1)-self.constant0)
            return kernel
    
        elif ktype == 'SE':
            m = torch.unsqueeze(m, 1)
            n = torch.unsqueeze(n, 0)
            norm = torch.norm(m-n, dim=-1)
            norm_squared = torch.pow(norm, 2)
            kernel = self.sigma1**2 * torch.exp(-0.5 * norm_squared / self.l1**2)
            return kernel
                
        
    def prediction(self, x, y, xs):
        kxx = self.kernel(x, x, self.ktype)
        kxxs = self.kernel(x, xs, self.ktype)
        kxsx = self.kernel(xs, x, self.ktype)
        kxsxs = self.kernel(xs, xs, self.ktype)
        
        psd_kernel = kxx + self.error * torch.eye(len(x)).to(device)
        
        u = torch.cholesky(psd_kernel)
        inverse = torch.cholesky_inverse(u)
        precompute = torch.mm(kxsx, inverse)  
        mu = torch.mm(precompute, y)
        covar = kxsxs - torch.mm(precompute, kxxs)
        
        boolean1 = True
        while boolean1:
            try:
                u = torch.cholesky(covar)
                boolean1 = False
            except RuntimeError:
                covar += self.error*torch.eye(len(xs)).to(device)

        y_dist = torch.distributions.normal.Normal(mu.view(-1), covar.diag())
        return y_dist
    
    
    def forward(self, x, y):
        kxx = self.kernel(x, x, self.ktype)
        
        mu = torch.Tensor([0]*len(x)).to(device)
        covar = kxx + self.error * torch.eye(len(x)).to(device)
         
        y_dist = torch.distributions.normal.Normal(mu.view(-1), covar.diag())
 
        loss = -y_dist.log_prob(y.view(-1)).mean()
        return loss
    
