
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
import time
import torch
import torch.nn as nn
import math
from utility import timer

def make_model(args, parent=False):
    return MetaCNN(args)

class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3): # kernel_size change from 3 to 5. 
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(3,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):
        output = self.meta_block(x)
        return output

class MetaCNN(nn.Module):
    def __init__(self, args):
        super(MetaCNN, self).__init__()
        r = args.scale[0]
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        G0 = args.G0 # output channels.64
        kSize = args.RDNkSize  # 
        kSize = 5

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.Feature =  nn.Sequential(
            nn.Conv2d(args.n_colors, G0, kSize, padding=kSize// 2, stride=1),
            nn.ReLU()
        )

        ## position to weight
        self.P2W = Pos2Weight(inC=G0)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat):
        #d1 =time.time()
        x = self.Feature(x)

        #self.wp_upscale.tic()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        #print(d2)
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW)

        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()

        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out = self.add_mean(out)
        #self.wp_upscale.hold()
        #print(f"feature learning : {self.fl.acc: .4f}, wp and upscale: {self.wp_upscale.acc:.4f}")

        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale[scale_idx]
    





