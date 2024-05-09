import torch
from torch import nn
import torch.nn.functional as F


class local_CNN(nn.Module):

    def __init__(self, resnet):
        super().__init__()  # Call the parent class's __init__() method
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer = resnet.layer1
        self.patching = nn.Unfold((4, 4), stride=(4,4))
        self.scale_down = nn.Linear(1024, 768)

        del resnet  # Free up memory (when Garbage Collector runs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer(x)
        x = torch.permute(self.patching(x), (0, 2, 1))  # Batch Size x 36 (Num tokens) x Dim (16384)
        x = self.scale_down(x)

        cls_embedding = torch.mean(x, dim = 1).unsqueeze(1)
        x = torch.cat((cls_embedding, x), dim= 1)

        return x