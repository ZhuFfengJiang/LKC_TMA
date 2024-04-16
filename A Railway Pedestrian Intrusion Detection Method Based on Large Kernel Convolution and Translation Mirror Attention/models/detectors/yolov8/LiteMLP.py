import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions. Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
 
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

if __name__ == '__main__':
    import time
    from thop import profile
    model= Mlp(in_features=3)
    x = torch.randn(1, 3, 960, 960)

    outputs = model(x)
    print(outputs.shape)
    # print("torch.Size([1, 256, 120, 120]) \n torch.Size([1, 512, 60, 60]) \n torch.Size([1, 512, 30, 30])")
    '''torch.Size([1, 64, 120, 120])
    torch.Size([1, 128, 60, 60])
    torch.Size([1, 256, 30, 30])
    ==============================
    ==============================
    GFLOPs : 6.96
    Params : 1.11 M
    [64, 128, 256]'''