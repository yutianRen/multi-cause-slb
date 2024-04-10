import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class mlpLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(mlpLayer, self).__init__()
        self.act = nn.GELU()
        self.layer = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.dropout(x)
        x = self.layer(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_feature, out_class):
        super(MLP, self).__init__()
        # multi cause: 32, 64, 128, 256, 512, 256, 128, 64, 32
        self.bn = nn.BatchNorm1d(in_feature)
        self.input_layer = nn.Linear(in_feature, 32)
        self.layer_list = nn.ModuleList([
                                         mlpLayer(32, 64, dropout=0.1),
                                         mlpLayer(64, 128),
                                         mlpLayer(128, 256),
                                         mlpLayer(256, 128),
                                         mlpLayer(128, 64, dropout=0.1),
                                         mlpLayer(64, 32, dropout=0.1),
                                        ])

        self.out_fc = nn.Linear(32, out_class)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        N, C = x.shape

        x = self.bn(x)
        x = self.input_layer(x)
        for layer in self.layer_list:
            x = layer(x)
        x = x.reshape((N, -1))

        out = self.out_fc(x)

        return out


class build_mlp:
    def __init__(self, in_feature=17, out_class=8):
        self.in_feature = in_feature
        self.out_class = out_class
    def build(self, num_classes):
        return MLP(self.in_feature, num_classes)

if __name__ == '__main__':
    model = MLP(5, 3)
    data = torch.ones(32, 10, 5)
    out = model(data)
    print(out.shape)
