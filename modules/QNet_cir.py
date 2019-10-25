import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, deque

eps = 1e-5

# -------------
# Single Layer
# -------------


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def center_crop(x):
    """
    center crop layer. crop [1:-2] to eliminate padding influence.
    Crop 1 element around the tensor
    input x can be a Variable or Tensor
    """
    return x[:, :, 1:-1, 1:-1].contiguous()


def center_crop7(x):
    """
    Center crop layer for stage1 of resnet. (7*7)
    input x can be a Variable or Tensor
    """

    return x[:, :, 2:-2, 2:-2].contiguous()


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# -------------------------------
# Several Kinds Bottleneck Blocks
# -------------------------------
class Bottleneck_CI(nn.Module):
    """
    Bottleneck with center crop layer, utilized in CVPR2019 model
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 1
        if abs(dilation - 2) < eps: padding = 2
        if abs(dilation - 3) < eps: padding = 3

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.last_relu:         # remove relu for the last block
            out = self.relu(out)

        out = center_crop(out)     # in-residual crop

        return out


class Bottleneck_BIG_CI(nn.Module):
    """
    Bottleneck with center crop layer, double channels in 3*3 conv layer in shortcut branch
    """
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None, dilation=1):
        super(Bottleneck_BIG_CI, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        padding = 1
        if abs(dilation - 2) < eps: padding = 2
        if abs(dilation - 3) < eps: padding = 3

        self.conv2 = nn.Conv2d(planes, planes*2, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.last_relu:  # feature out no relu
            out = self.relu(out)

        out = center_crop(out)  # in-layer crop

        return out


# ---------------------
# ResNeXt Tools
# ---------------------

class ResNet(nn.Module):
    """
    ResNet with 22 layer utilized in CVPR2019 paper.
    Usage: ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True], 64, [64, 128])
    """

    def __init__(self, block, layers, last_relus, s2p_flags, firstchannels=64, channels=[64, 128], dilation=1):
        self.inplanes = firstchannels
        self.stage_len = len(layers)
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, firstchannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # stage2
        if s2p_flags[0]:
            self.layer1 = self._make_layer(block, channels[0], layers[0], stride2pool=True, last_relu=last_relus[0])
        else:
            self.layer1 = self._make_layer(block, channels[0], layers[0], last_relu=last_relus[0])

        # stage3
        if s2p_flags[1]:
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride2pool=True, last_relu=last_relus[1], dilation=dilation)
        else:
            self.layer2 = self._make_layer(block, channels[1], layers[1], last_relu=last_relus[1], dilation=dilation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, last_relu, stride=1, stride2pool=False, dilation=1):
        """
        :param block:
        :param planes:
        :param blocks:
        :param stride:
        :param stride2pool: translate (3,2) conv to (3, 1)conv + (2, 2)pool
        :return:
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample, dilation=dilation))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(self.inplanes, planes, last_relu=last_relu, dilation=dilation))
            else:
                layers.append(block(self.inplanes, planes, last_relu=True, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)     # stride = 2
        x = self.bn1(x)
        x = self.relu(x)
        x = center_crop7(x)
        x = self.maxpool(x)   # stride = 4

        x = self.layer1(x)
        x = self.layer2(x)    # stride = 8

        return x



# --------------
# ResNeXt Tools
# --------------
class BasicBlock_C(nn.Module):
    """
    increasing cardinality is a more effective way of
    gaining accuracy than going deeper or wider
    """

    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, expansion=2, last_relu=True):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0',
                 nn.Conv2d(inner_width, inner_width, 3, stride=1, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inner_width * self.expansion))
            ]
        ))
        self.shortcut = nn.Sequential()
        if in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=1, bias=False)
            )
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)
        self.last_relu = last_relu

    def forward(self, x):
        out = self.basic(x)
        out += self.shortcut(x)

        if self.last_relu:
            return center_crop(F.relu(self.bn0(out)))    # CIR
        else:
            return center_crop(self.bn0(out))


class ResNeXt(nn.Module):
    """
    ResNeXt with 22 layer utilized in CVPR2019 paper.
    Usage: ResNeXt([3, 4], 32, 4)
    """
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=0)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(num_blocks[0],  last_relu=True)
        self.layer2 = self._make_layer(num_blocks[1],  last_relu=False, stride2pool=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)

    def _make_layer(self, num_blocks, last_relu=True, stride2pool=False):

        layers = []
        for i in range(0, num_blocks):
            if i == num_blocks - 1:
                layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, self.expansion, last_relu=last_relu))
            else:
                layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, self.expansion))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality

            if i == 0 and stride2pool:
                layers.append(self.maxpool)

        self.bottleneck_width *= 2

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.maxpool1(out)

        out = self.layer1(out)
        out = self.layer2(out)

        return out

class ResNet22(nn.Module):
    """
    default: unfix gradually (lr: 1r-2 ~ 1e-5)
    optional: unfix all at first with small lr (lr: 1e-7 ~ 1e-3)
    """
    def __init__(self):
        super(ResNet22, self).__init__()
        self.features = ResNet(Bottleneck_CI, [3, 4], [True, False], [False, True])
        self.feature_size = 512
        self.train_num = 0
        self.unfix(0.0)

    def forward(self, x):
        x = self.features(x)
        return x

    def unfix(self, ratio):
        """
        unfix gradually as paper said
        """
        if abs(ratio - 0.0) < eps:
            self.train_num = 2  # epoch0 1*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.1) < eps:
            self.train_num = 3  # epoch5 2*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.2) < eps:
            self.train_num = 4  # epoch10 3*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.3) < eps:
            self.train_num = 6  # epoch15 4*[1,3,1]  stride2pool makes stage2 have a more index
            self.unlock()
            return True
        elif abs(ratio - 0.5) < eps:
            self.train_num = 7  # epoch25 5*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.6) < eps:
            self.train_num = 8  # epoch30 6*[1,3,1]
            self.unlock()
            return True
        elif abs(ratio - 0.7) < eps:
            self.train_num = 9  # epoch35 7*[1,3,1]
            self.unlock()
            return True

        return False

    def unlock(self):
        for p in self.parameters():
            p.requires_grad = False

        for i in range(1, self.train_num):  # zzp pay attention here
            if i <= 5:
                m = self.features.layer2[-i]
            elif i <= 8:
                m = self.features.layer1[-(i - 5)]
            else:
                m = self.features

            for p in m.parameters():
                p.requires_grad = True
        self.eval()
        self.train()

    def train(self, mode=True):
        self.training = mode
        if mode == False:
            super(ResNet22, self).train(False)
        else:
            for i in range(self.train_num):  # zzp pay attention here
                if i <= 5:
                    m = self.features.layer2[-i]
                elif i <= 8:
                    m = self.features.layer1[-(i - 5)]
                else:
                    m = self.features
                m.train(mode)

        return self

class QNet_cir(nn.Module):
    def __init__(self):
        super(QNet_cir, self).__init__()
        self.features = ResNet22()
        self.fc1 = nn.Linear(2055, 512)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(512, 2)



    def forward(self, x, a):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x,a], dim=1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x

def QNet_train(q, q_target, memory, optimizer):
    for i in range(5):
        s, a, r, s_prime = memory.sample(32)
        a2 = torch.zeros(7)
        mq = 0
        for j in range(7):
            a2[j] = 1
            q_out = q(s_prime, a2)
            if q_out[0] > mq:
                mq = q_out[0]
        target = r + 0.98 * mq
        q_a = q(s, a)[0]
        loss = F.smooth_l1_loss(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen = 1000)


    def put(self, transistion):
        self.buffer.append(transistion)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)

        return torch.tensor(s_lst, dtype=torch.float).cuda(), torch.tensor(a_lst).cuda(), \
               torch.tensor(r_lst).cuda(), torch.tensor(s_prime_lst, dtype=torch.float).cuda()

    def size(self):
        return len(self.buffer)

if __name__ == '__main__':
    net = QNet_cir()
    # print(net)
    x = torch.rand(2,3,107,107)
    a = torch.rand(2,7)
    net.eval()
    out = net(x,a)
    print(out.shape)


