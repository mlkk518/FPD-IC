import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange
from typing import Optional, Any

from .distributions import DiagonalGaussianDistribution
from .config import Config, AttnMode


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        print(f"building AttnBlock (vanilla) with {in_channels} in_channels")

        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        print(
            f"building MemoryEfficientAttnBlock (xformers) with {in_channels} in_channels"
        )
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = Config.xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


class SDPAttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        print(f"building SDPAttnBlock (sdp) with {in_channels} in_channels")
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = F.scaled_dot_product_attention(q, k, v)

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        out = rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)
        out = self.proj_out(out)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in [
        "vanilla",
        "sdp",
        "xformers",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "sdp":
        return SDPAttnBlock(in_channels)
    elif attn_type == "xformers":
        return MemoryEfficientAttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()




# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = self.shortcut(x)
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out += residual
#         out = self.relu(out)
#         return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1.2, stride=1):
        """
        优化版残差块 (支持通道扩展和自适应调节)

        参数：
        expansion : 通道扩展系数 (1.0-2.0)
        stride    : 空间下采样步长 (1或2)
        """
        super().__init__()
        expanded_channels = int(out_channels * expansion)

        # 主路径 (带通道扩展)
        self.conv1 = nn.Conv2d(in_channels, expanded_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(expanded_channels)

        self.conv2 = nn.Conv2d(expanded_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)

        # # 自适应通道门控
        # self.attention = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(out_channels, out_channels // 8, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels // 8, out_channels, 1),
        #     nn.Sigmoid()
        # )

        # 短路连接优化
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channels)
            )

        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        """He初始化 + 最后一层零初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # 最后一个卷积层零初始化
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x):
        # 主路径
        residual = self.shortcut(x)

        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)

        # 通道注意力机制
        # attention_mask = self.attention(out)
        # out = out * attention_mask * 0.5  # 衰减因子防止过调节

        # 残差连接
        out += residual
        return F.relu(out, inplace=True)
class Adjust_block(nn.Module):
    def __init__(self, in_ch, emb_dim=32):
        super().__init__()


        # self.encode_head = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1)

        self.encode_head = ResidualBlock(in_ch * 2, emb_dim)

        self.scale = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(emb_dim, in_ch, kernel_size=3, padding=1),
        )

        self.shift = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(emb_dim, in_ch, kernel_size=3, padding=1),
        )

    def forward(self, dec_feat, enc_feat, weight=1.0):

        prior_feat = self.encode_head(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(prior_feat)
        shift = self.shift(prior_feat)


        # if isinstance(weight, (int, float)):
        #     weight = torch.full_like(enc_feat, weight, dtype=torch.float32)

        residual = weight * (enc_feat * (1.0 + scale) + shift)
        dec_feat_new = dec_feat + residual

        return dec_feat_new


# class Feat_Adaptor(nn.Module):
#     def __init__(self, multi_dims, alpha=1.0, depth=5):
#         super().__init__()
#
#         self.Adjust_blocks = nn.ModuleList([
#             Adjust_block(multi_dims[i], alpha) for i in range(depth)
#         ])
#
#     def forward(self, enc_feat, dec_feat):
#         dec_feat_new = []
#         for i in range(len(enc_feat)):
#             dec_feat_hat = self.Adjust_blocks[i](enc_feat[i], dec_feat[i])
#             dec_feat_new.append(dec_feat_hat)
#
#         return dec_feat_hat
class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        Adaptor_Flag=False,
        alpha=1.0,
        **ignore_kwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, Stage2=False):
        # timestep embedding
        temb = None
        h_list = []
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if Stage2:
            hs_selected = [x, hs[2], hs[5], hs[8], hs[11]]
            return h, hs_selected  ## 128-256, 256-128, 512-64, 512-32

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        Adaptor_Flag=False,
        alpha=1.0,
        **ignorekwargs,
    ):
        super().__init__()
        ### setup attention type
        if Config.attn_mode == AttnMode.SDP:
            attn_type = "sdp"
        elif Config.attn_mode == AttnMode.XFORMERS:
            attn_type = "xformers"
        else:
            attn_type = "vanilla"
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.Adaptor_Flag = Adaptor_Flag

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )
        if Adaptor_Flag:
            self.Adaptor_Flag = Adaptor_Flag
            self.alpha = alpha
            # multi_dims = [3, 128, 256, 512, 512]
            # multi_dims = [3]
            # self.Adjust_blocks = nn.ModuleList([
            #     Adjust_block(multi_dims[i]) for i in range(len(multi_dims))
            # ])
            self.Adjust_blocks = Adjust_block(in_ch=3, emb_dim=64)

    def forward(self, z, enc_feats=None, weight=1.0):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        ## 128-256, 256-128, 512-64, 512-32
        # for i in range(len(enc_feats)):
        #     out = enc_feats[i]
        #     print("out shape", i, out.shape)


        # z to block_in
        h = self.conv_in(z)
        # if enc_feats:
        #     index = 4
        #     # h = self.Adjust_blocks[index](h, enc_feats[index], weight=1)
        #     index -= 1

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)  ##  512*32*32
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

                # if enc_feats and i_level == 2 and i_block == 0:
                #     # h = self.Adjust_blocks[index](h, enc_feats[index], weight=1)
                #     index -= 1
                # elif enc_feats and i_level == 1 and i_block == 0:
                #     # h = self.Adjust_blocks[index](h, enc_feats[index], weight=1)
                #     index -= 1
                # elif enc_feats and i_level == 0 and i_block == 0:
                #     # h = self.Adjust_blocks[index](h, enc_feats[index], weight=1)
                #     index -= 1

            if i_level != 0:
                h = self.up[i_level].upsample(h)
        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)



        if enc_feats:
            if self.alpha != 1.0:  ## 由于weight  默认为1, 所以无需进行调节
                weight = self.alpha
            h = self.Adjust_blocks(h, enc_feats[0], weight=weight)

        return h


class AutoencoderKL(nn.Module):

    def __init__(self, ddconfig, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim


    def encode(self, x, Stage2=False):

        if Stage2:
            h, enc_feats = self.encoder(x, Stage2)
        else:
            h = self.encoder(x, Stage2)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if Stage2:
            return posterior, enc_feats
        return posterior

    def decode(self, z, enc_feats, weight):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, enc_feats=enc_feats, weight=weight)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior
