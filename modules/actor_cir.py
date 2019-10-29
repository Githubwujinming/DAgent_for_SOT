import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from modules.QNet_cir import ResNet22

np.random.seed(122)
torch.manual_seed(455)
torch.cuda.manual_seed(788)

class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x

from collections import OrderedDict
class Actor(nn.Module):
    def __init__(self, model_path=None):
        super(Actor, self).__init__()
        self.features = ResNet22()
        self.fc1 = nn.Linear(4096, 512)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(512, 3)
        self.out = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, xl, xg):
        x = self.features(xl)
        y = self.features(xg)
        x = torch.cat([x, y], dim=1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.out(x)

        return x




if __name__ == '__main__':
    net = Actor()
    # print(net)
    net = net.cuda()
    z = torch.rand(2,3,107,107).cuda()
    x = torch.rand(2,3,107,107).cuda()
    net.eval()
    out = net(z, x)
    print(out)

