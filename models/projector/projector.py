import torch
from torch import nn
from utils.utils import trunc_normal_
import torchvision



class MyModel(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head , CNN_head = None):
        super(MyModel, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        # self.CNN_head = CNN_head
        self.backbone = backbone
        self.head = head

    def forward(self, x , attention_map = False ,  origin_rgb = None , origin_ir = None):

        _out = self.backbone(x)
        if isinstance(_out,tuple):
            _out = _out[1] # batch shape : (bs,num_patches,features)
        return _out[0].reshape(_out[0].size(0),12,-1),self.head(_out)



class PatchProjector(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=2, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        # nlayers = max(nlayers, 1)
        # if nlayers == 1:
        #     self.mlp = nn.Linear(in_dim, bottleneck_dim)
        # else:
        #     layers = [nn.Linear(in_dim, hidden_dim)]
        #     if use_bn:
        #         layers.append(nn.BatchNorm1d(hidden_dim))
        #     layers.append(nn.GELU())
        #     for _ in range(nlayers - 2):
        #         layers.append(nn.Linear(hidden_dim, hidden_dim))
        #         if use_bn:
        #             layers.append(nn.BatchNorm1d(hidden_dim))
        #         layers.append(nn.GELU())
        #     layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        #     self.mlp = nn.Sequential(*layers)
        # self.apply(self._init_weights)
        # self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape : (batch , patch_nums , patch_represent_dim)
        # x = self.mlp(x)
        # x = nn.functional.normalize(x, dim=-1, p=2)
        # x = self.last_layer(x)
        return x
