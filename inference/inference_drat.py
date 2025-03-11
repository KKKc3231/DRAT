import argparse
import cv2
import glob
import numpy as np
import os
import torch

import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from archs.arch_util import default_init_weights, make_layer, pixel_unshuffle
from add.encoder import Encoder
from archs.arch_util import Upsample
import torch.nn as nn
import torchvision.models
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# load dual-encoder for degradation representation
encoder_aux = Encoder()
encoder_blur = Encoder()

encoder_aux.load_state_dict(torch.load("/data/chensen/code/BasicSR/checkpoint/Dual-encoder/encoder_hr.pt")) # 
encoder_blur.load_state_dict(torch.load("/data/chensen/code/BasicSR/checkpoint/Dual-encoder/encoder_x2_x4_div2krk.pt"))

encoder_aux.to('cuda')
encoder_blur.to('cuda')

encoder_aux.eval()
encoder_blur.eval()

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Conv_PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1, 0),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(inner_dim, dim, 1, 1, 0),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult=1, bias=False, dropout=0.):
        super().__init__()

        hidden_features = int(dim * mult)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# MBConv

class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout=0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x


class Dropsample(nn.Module):
    def __init__(self, prob=0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device=device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)


def MBConv(
        dim_in,
        dim_out,
        *,
        downsample,
        expansion_rate=4,
        shrinkage_rate=0.25,
        dropout=0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim),
        # nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate=shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        # nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout=dropout)

    return net


# attention related classes
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        # relative positional bias
        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h=h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        if self.with_pe:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=window_height, w2=window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x=height, y=width)


class Block_Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=32,
            bias=False,
            dropout=0.,
            window_size=7,
            with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.ps = window_size
        self.scale = dim_head ** -0.5
        self.with_pe = with_pe

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        # project for queries, keys, values
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h=self.heads, w1=self.ps,
                                          w2=self.ps), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, '(b x y) head (w1 w2) d -> b (head d) (x w1) (y w2)', x=h // self.ps, y=w // self.ps,
                        head=self.heads, w1=self.ps, w2=self.ps)

        out = self.to_out(out)
        return out


class Channel_Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps,
                                head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h // self.ps, w=w // self.ps,
                        ph=self.ps, pw=self.ps, head=self.heads)

        out = self.project_out(out)

        return out


class Channel_Attention_grid(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            bias=False,
            dropout=0.,
            window_size=7
    ):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1)

        q, k, v = map(
            lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)', ph=self.ps, pw=self.ps,
                                head=self.heads), qkv)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)', h=h // self.ps, w=w // self.ps,
                        ph=self.ps, pw=self.ps, head=self.heads)

        out = self.project_out(out)

        return out


class GDA_Block(nn.Module):
    def __init__(self, channel_num=64, bias=True, ffn_bias=True, window_size=1    , with_pe=False, dropout=0.0):
        super(GDA_Block, self).__init__()

        w = window_size

        self.layer = nn.Sequential(
            MBConv(
                channel_num,
                channel_num,
                downsample=False,
                expansion_rate=1,
                shrinkage_rate=0.25
            ),

            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),  # block-like attention
            PreNormResidual(channel_num, Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                   window_size=window_size, with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num,
                                 Channel_Attention(dim=channel_num, heads=4, dropout=dropout, window_size=window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),  # grid-like attention
            PreNormResidual(channel_num, Attention(dim=channel_num, dim_head=channel_num // 4, dropout=dropout,
                                                   window_size=window_size, with_pe=with_pe)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim=channel_num, heads=4, dropout=dropout,
                                                                     window_size=window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class DRT(nn.Module):

    def __init__(self, degradation_dim=64, num_feat=64):
        super(DRT, self).__init__()
        degradation_dim= 64 #256 #64
        self.fc = nn.Sequential(
            nn.Linear(degradation_dim, (degradation_dim + num_feat * 2) // 2),
            nn.ReLU(True),
            nn.Linear((degradation_dim + num_feat * 2) // 2, (degradation_dim + num_feat * 2) // 2),
            nn.ReLU(True),
            nn.Linear((degradation_dim + num_feat * 2) // 2, num_feat * 2),
        )
        default_init_weights([self.fc], 0.1)

    def forward(self, x, d):
        d = self.fc(d)
        d = d.view(d.size(0), d.size(1), 1, 1)
        gamma, beta = torch.chunk(d, chunks=2, dim=1)
        return (1 + gamma) * x + beta

# enhance channel attention
class ECA_layer(nn.Module):
    """Constructs a ECA module.
        Args:
            channel: Number of channels of the input feature map
            k_size: Adaptive selection of kernel size
    """
    def __init__(self,channel,k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=k_size,padding=(k_size-1)//2,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        """
        x: feture map: B * C * H * W
        dea: degradation representation: B * C
        """
        x1 = x[0]
        dea = x[1]
        dea = dea[:,:,None,None] # the same dim
        att = self.avg_pool(dea)
        att1 = self.conv(att.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        att2 = self.sigmoid(att1)
        return x1 * att2

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self,num_feat=32, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat*2, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        # channel attention
        # self.fea = fea

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * 3 * 3, bias=False)
        )

        self.conv = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,padding=(1//2),bias=True)
        self.eca = ECA_layer(channel=num_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.osa = GDA_Block()
        self.aff = DRT()
        
        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        b,c,h,w = x[0].size()
        kernel = self.kernel(x[1]).view(-1,1,3,3)
        x0 = self.lrelu(F.conv2d(x[0].view(1,-1,h,w),kernel,groups=b*c,padding=(3-1)//2))
        x0 = self.conv(x0.view(b,-1,h,w))
        x_eca = self.eca(x)
        x_scda = self.osa(x_eca)

        x1 = self.lrelu(self.conv1(torch.cat((x0, x_gda), 1)))
        x2 = self.lrelu(self.conv2(torch.cat((x_gda, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x_gda, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x_gda, x1, x2 , x3), 1)))
        out = self.lrelu(self.conv5(torch.cat((x_gda, x1, x2, x3, x4), 1)))
        out = self.aff(out,x[1])
        
        return (out * 0.2) + x[0]

class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        # self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        # self.ca = CA_layer(num_feat,num_feat,reduction=8)

    def forward(self, x):
        """
        parameters:
        x: input + degradation
        x[0]: input
        x[1]: degradation_ori
        x[2]: degradation_com
        """
        out = self.rdb1(x)
        out1 = [out,x[1]]
        out2 = self.rdb2(out1)
        return [out2 * 0.2 + x[0],x[1]]


class DRATNet(nn.Module):
    """
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 10
        num_grow_ch (int): Channels for each growth. Default: 16.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(DRATNet, self).__init__()
        
        self.scale = scale
        # feature dim 256 -> 64/32
        self.compress1 = nn.Linear(256,128)
        self.compress2 = nn.Linear(128,64)
   
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch,)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # upsample
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # pixelshuffle
        self.upsample = Upsample(scale, num_feat)

    def forward(self, x):

        fea_blur = encoder_blur(x)
        fea_aux = encoder_aux(x)
        fea_ori = fea_blur - fea_aux
        
        fea1 = self.compress1(fea_ori)
        fea2 = self.compress2(fea1)
        
        feat = x
        feat = self.conv_first(feat)
        
        input = [feat,fea2]
        out = self.body(input)
        new_input = out[0]
        body_feat = self.conv_body(new_input)
        feat = feat + body_feat
        
        # upsample
        feat = self.lrelu(self.upsample(feat))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        "/data/chensen/code/BasicSR/checkpoint/DRAT/setting2/DRAT_div2krk_x4.pth"  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='/data/chensen/DRAT_inference/input/', help='input test image folder')
    parser.add_argument('--output', type=str, default='/data/chensen/DRAT_inference/output/', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = DRANNet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=10, num_grow_ch=24)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=False)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)
        # read image
        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)
        # inference
        try:
            with torch.no_grad():
                output = model(img)
        except Exception as error:
            print('Error', error, imgname)
        else:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.output, f'{imgname}_DRAT.png'), output)


if __name__ == '__main__':
    main()
