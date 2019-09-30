import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_, zeros_, normal_

gamma = 0.98
def weights_init(model):

    if isinstance(model, nn.Conv2d):
        xavier_normal_(model.weight, gain=math.sqrt(2.0))
        constant_(model.bias, 0.1)

    elif isinstance(model, nn.BatchNorm2d):
        normal_(model.weight, 1.0, 0.02)
        zeros_(model.bias)


class T_Policy(nn.Module):
    def __init__(self, T, learnning_rate=0.0002):
        super().__init__()
        self.data = []
        self.conv = nn.Sequential(
            nn.Conv2d(T, 32, kernel_size=3,
                      stride=2, groups=1,
                      bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 32, kernel_size=2,
                      stride=2, groups=1,
                      bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128, T),
            nn.Dropout(0.5))
        self.optimizer = optim.Adam(self.parameters(), lr=learnning_rate)

    def forward(self, x):
        output = self.conv(x)
        output = output.view(-1, 512)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.softmax(output, dim=1)
        return output

    def put_data(self, item):
        self.data.append(item)



    def train_policy(self):
        R = 0
        for r, log_prob in self.data[::-1]:
            R = r + gamma * R
            loss = -log_prob * (R - 0.3)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.data = []


#
# #
# if __name__ == '__main__':
#     net = SiameseNet(BaselineEmbeddingNet())
#     # print(net)
#     net = net.cuda()
#     weights_init(net)
#     z = torch.rand(2,3,127,127).cuda()
#     x = torch.rand(2,3,255,255).cuda()
#     net.eval()
#     out = net(z, x).permute(1, 0, 2, 3)
#     print(out.shape)
#     policy = T_Policy(2).cuda()
#     weights_init(policy)
#     out = policy(out)
#     print(out)