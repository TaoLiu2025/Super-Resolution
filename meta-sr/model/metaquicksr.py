
# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
import time
import torch
import torch.nn as nn
import math
from utility import timer

def make_model(args, parent=False):
    return MetaQuickSR(args)

class Pos2Weight(nn.Module): 
    def __init__(self,inC, kernel_size=3, outC=3): # change kernel size from 3 to 5.
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
        #print(f"output shape {output.shape}")
        return output

class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2
    
class MetaQuickSR(nn.Module):
    def __init__(self, args):
        super(MetaQuickSR, self).__init__()
        r = args.scale[0]
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        num_intermediate_layers = 3
        num_channels = 16
        self.fl_time = 0

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend([
                nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
                nn.ReLU()
            ])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=args.n_colors, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            *intermediate_layers,
        )

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.P2W = Pos2Weight(inC=num_channels)

    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)
       
        return x.contiguous().view(-1, C, H, W)


    def forward(self, x, pos_mat):
        #print(f"\n feature : {self.fl_time:.4f}")
        # d1 =time.time()
        x = self.cnn(x)      
        # d2 = time.time()
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        # d3 = time.time()
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW) 

        # cols = nn.functional.unfold(up_x, 5,padding=2)
        cols = nn.functional.unfold(up_x, 3,padding=1)
        
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()
       
        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)
  

        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out = self.add_mean(out)
        # d4 = time.time()
        #print(f"feature learning: {(d2 - d1):.5f}, weight prediction: {(d3 - d2):.5f}. feature mapping: {(d4 - d3):.5f}")
        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        self.scale = self.args.scale[scale_idx]
    





