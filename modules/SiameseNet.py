import torch.nn as nn
import torch.nn.functional as F



class SiameseNet(nn.Module):

    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net
        self.match_batchnorm = nn.BatchNorm2d(1)


    def forward(self, x1, x2):
        embedding_reference = self.embedding_net(x1)
        embedding_search = self.embedding_net(x2)
        match_map = self.match_corr(embedding_reference, embedding_search)
        return match_map

    def match_corr(self, embed_ref, embed_srch):

        b, c, h, w = embed_srch.shape
        match_map = F.conv2d(embed_srch.view(1, b * c, h, w),
                             embed_ref, groups = b)
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_batchnorm(match_map)

        return match_map