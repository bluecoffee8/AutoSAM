import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.data.size(0),-1)



class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1',ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2',DWConvLayer(out_channels, out_channels, stride=stride))
        
    def forward(self, x):
        return super().forward(x)

class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels,  stride=1,  bias=False):
        super().__init__()
        out_ch = out_channels
        
        groups = in_channels
        kernel = 3
        #print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')
        
        self.add_module('dwconv', nn.Conv2d(groups, groups, kernel_size=3,
                                          stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(groups))
    def forward(self, x):
        return super().forward(x)  

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        #print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', nn.Conv2d(in_channels, out_ch, kernel_size=kernel,          
                                          stride=stride, padding=kernel//2, groups=groups, bias=bias))
        self.add_module('norm', nn.BatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))                                          
    def forward(self, x):
        return super().forward(x)

"""
Upsample Layer is novel addition to original HardNet architecture from AutoSAM paper by Shaharabany et al (2023). 

The code here is from UNet architecture: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
"""
# beginning of additional code

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
        # self.double_conv = nn.Sequential(
        #    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), 
        #    nn.ReLU(inplace=True), 
        #    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), 
        #    nn.BatchNorm2d(out_channels),
        #    nn.Tanh(inplace=True)
        # )
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh(inplace=True)

    def forward(self, x):
        # return self.double_conv(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = self.tanh(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

"""Decoder network for HarDNet as described in AutoSAM by Shaharabany et al (2023), using HarDNet85 architecture"""
class HarDNetDecoder(nn.Module):
  def __init__(self, channels=[1280, 720, 480, 320, 256], bilinear=True):
    super().__init__()
    self.conv11 = nn.Conv2d(2 * channels[0], channels[1], kernel_size=3, padding=1, bias=False)
    self.relu1 = nn.ReLU(inplace=True)
    self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv12 = nn.Conv2d(2 * channels[1], channels[2], kernel_size=3, padding=1, bias=False)
    self.batchnorm1 = nn.BatchNorm2d(channels[2])
    self.tanh1 = nn.Tanh()
    
    self.conv21 = nn.Conv2d(2 * channels[2], channels[3], kernel_size=3, padding=1, bias=False)
    self.relu2 = nn.ReLU(inplace=True)
    self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.conv22 = nn.Conv2d(2 * channels[3], channels[4], kernel_size=3, padding=1, bias=False) 
    self.batchnorm2 = nn.BatchNorm2d(channels[4]) 
    self.tanh2 = nn.Tanh()
    
  def forward(self, x, x1, x2, x3, x4):
    x = torch.cat([x, x4], dim=1)
    x = self.conv11(x)
    x = self.relu1(x)
    x = self.up1(x)
    x = torch.cat([x, x3], dim=1)
    x = self.conv12(x)
    x = self.batchnorm1(x)
    x = self.tanh1(x)
    x = torch.cat([x, x2], dim=1)
    x = self.conv21(x)
    x = self.relu2(x)
    x = self.up2(x)
    x = torch.cat([x, x1], dim=1)
    x = self.conv22(x)
    x = self.batchnorm2(x)
    x = self.tanh2(x)
    return x


# end of additional code

class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
          return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
          dv = 2 ** i
          if layer % dv == 0:
            k = layer - dv
            link.append(k)
            if i > 0:
                out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
          ch,_,_ = self.get_link(i, base_ch, growth_rate, grmul)
          in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0 # if upsample else in_channels
        for i in range(n_layers):
          outch, inch, link = self.get_link(i+1, in_channels, growth_rate, grmul)
          self.links.append(link)
          use_relu = residual_out
          if dwconv:
            layers_.append(CombConvLayer(inch, outch))
          else:
            layers_.append(ConvLayer(inch, outch))
          
          if (i % 2 == 0) or (i == n_layers - 1):
            self.out_channels += outch
        #print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)
        
    def forward(self, x):
        layers_ = [x]
        
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:            
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)
            
        t = len(layers_)
        out_ = []
        for i in range(t):
          if (i == 0 and self.keepBase) or \
             (i == t-1) or (i%2 == 1):
              out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out
        
        
        
        
class HarDNet(nn.Module):
    def __init__(self, depth_wise=False, arch=85, pretrained=True, weight_path=''):
        super().__init__()
        first_ch  = [32, 64]
        second_kernel = 3
        max_pool = True
        grmul = 1.7
        drop_rate = 0.1
        
        #HarDNet68
        ch_list = [  128, 256, 320, 640, 1024]
        gr       = [  14, 16, 20, 40,160]
        n_layers = [   8, 16, 16, 16,  4]
        downSamp = [   1,  0,  1,  1,  0]
        
        if arch==85:
          #HarDNet85
          first_ch  = [48, 96]
          ch_list = [  192, 256, 320, 480, 720, 1280]
          gr       = [  24,  24,  28,  36,  48, 256]
          n_layers = [   8,  16,  16,  16,  16,   4]
          downSamp = [   1,   0,   1,   0,   1,   0]
          drop_rate = 0.2
        elif arch==39:
          #HarDNet39
          first_ch  = [24, 48]
          ch_list = [  96, 320, 640, 1024]
          grmul = 1.6
          gr       = [  16,  20, 64, 160]
          n_layers = [   4,  16,  8,   4]
          downSamp = [   1,   1,  1,   0]
          
        if depth_wise:
          second_kernel = 1
          max_pool = False
          drop_rate = 0.05
        
        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append (
             ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                       stride=2,  bias=False) )
  
        # Second Layer
        self.base.append ( ConvLayer(first_ch[0], first_ch[1],  kernel=second_kernel) )
        
        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
          self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
          self.base.append ( DWConvLayer(first_ch[1], first_ch[1], stride=2) )

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append ( blk )
            
            if i == blks-1 and arch == 85:
                self.base.append ( nn.Dropout(0.1))
            
            self.base.append ( ConvLayer(ch, ch_list[i], kernel=1) )
            ch = ch_list[i]
            if downSamp[i] == 1:
              if max_pool:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
              else:
                self.base.append ( DWConvLayer(ch, ch, stride=2) )

        """
        Below code block is commented out since we apply 2 decoder blocks. 
        """
        # ch = ch_list[blks-1]
        # self.base.append (
        #     nn.Sequential(
        #         nn.AdaptiveAvgPool2d((1,1)),
        #         Flatten(),
        #         nn.Dropout(drop_rate),
        #         nn.Linear(ch, 1000) ))
                
        #print(self.base)
        
        if pretrained:
          if hasattr(torch, 'hub'):
          
            if arch == 68 and not depth_wise:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet68-5d684880.pth'
            elif arch == 85 and not depth_wise:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet85-a28faa00.pth'
            elif arch == 68 and depth_wise:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth'
            else:
              checkpoint = 'https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth'

            self.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False), strict=False) # we need strict=False since we do not consider last linear layer
          else:
            postfix = 'ds' if depth_wise else ''
            weight_file = '%shardnet%d%s.pth'%(weight_path, arch, postfix)            
            if not os.path.isfile(weight_file):
              print(weight_file,'is not found')
              exit(0)
            weights = torch.load(weight_file)
            self.load_state_dict(weights, strict=False) # we need strict=False
          
          postfix = 'DS' if depth_wise else ''
          print('ImageNet pretrained weights for HarDNet%d%s is loaded'%(arch, postfix))
        
        self.decoder = HarDNetDecoder()
          
    def forward(self, x):
        # print(x.shape)
        # print()
        for layer in self.base:
          x = layer(x)
          # print(layer)
          # print(x.shape)
          # print()
          if x.shape[-1] == 32 and x.shape[-2] == 32 and x.shape[-3] == 1280:
             x4 = x
          elif x.shape[-1] == 64 and x.shape[-2] == 64 and x.shape[-3] == 720:
             x3 = x
          elif x.shape[-1] == 64 and x.shape[-2] == 64 and x.shape[-3] == 480:
             x2 = x
          elif x.shape[-1] == 128 and x.shape[-2] == 128 and x.shape[-3] == 320:
             x1 = x
        
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)

        x = self.decoder(x, x1, x2, x3, x4)

        x = F.interpolate(
           x,
           (64, 64),
           mode="bilinear", 
           align_corners=False,
        )

        return x
        
        
        
        