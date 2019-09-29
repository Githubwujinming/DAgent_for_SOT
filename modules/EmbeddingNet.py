import torch
import torch.nn as nn


class BaselineEmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fully_conv = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11,
                                                    stride=2, groups=1,
                                                    bias=True),
                                          nn.BatchNorm2d(96),
                                          nn.ReLU(),
                                          nn.MaxPool2d(3, stride=2),

                                          nn.Conv2d(96, 256, kernel_size=5,
                                                    stride=1, groups=2,
                                                    bias=True),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.MaxPool2d(3, stride=1),
                                          nn.Conv2d(256, 384, kernel_size=3,
                                                    stride=1, groups=1,
                                                    bias=True),
                                          nn.BatchNorm2d(384),
                                          nn.ReLU(),

                                          nn.Conv2d(384, 384, kernel_size=3,
                                                    stride=1, groups=2,
                                                    bias=True),
                                          nn.BatchNorm2d(384),
                                          nn.ReLU(),

                                          nn.Conv2d(384, 32, kernel_size=3,
                                                    stride=1, groups=2,
                                                    bias=True))

    def forward(self, x):
        output = self.fully_conv(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)