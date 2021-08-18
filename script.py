import torch
import torch.nn as nn

feature = torch.zeros(4,5,3)
for i in range(4):
    for j in range(5):
        val = i*5 + j
        for k in range(3):
            feature[i, j, k] = val
print(feature)
feature = feature.view(-1, 3)
linear_0 = nn.Linear(3, 6, bias=True)
bn_0 = nn.BatchNorm1d(6)
feature = bn_0(linear_0(feature))
feature = feature.view(4,5,-1)
print(feature)
print(feature.shape)