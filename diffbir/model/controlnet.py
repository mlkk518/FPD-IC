import torch
import torch as th
import torch.nn as nn

from .util import conv_nd, linear, zero_module, timestep_embedding, exists, conv_2d_layers
from .attention import SpatialTransformer
from .unet import (
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
    UNetModel,
)
from collections import deque

class ControlledUnetModel(UNetModel):

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        **kwargs,
    ):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))

        enc_control = deque()
        dec_control = deque()
        # if control is not None:
        for ctrl in control:
            enc_feat, dec_feat = torch.chunk(ctrl, 2, dim=1)
            enc_control.append(enc_feat)
            dec_control.append(dec_feat)


        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
            h += enc_control.popleft()
        h = self.middle_block(h, emb, context)
        h += enc_control.popleft()

        # if control is not None:
        h += dec_control.pop()
            # scale, shift = torch.chunk(control.pop(), 2, dim=1)
            # h = h * (1.0 + scale) + shift

        for i, module in enumerate(self.output_blocks):
            # if only_mid_control or control is None:
            #     h = torch.cat([h, hs.pop()], dim=1)
            # else:
                # print("else  control  h.shape :", h.shape)
                # scale, shift = torch.chunk(control.pop(), 2, dim=1)
            h_con = hs.pop() + dec_control.pop()
            h = torch.cat([h, h_con], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class DownsampleModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, edim=32):
        super(DownsampleModel, self).__init__()

        # 第一次下采样 + 残差连接
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, edim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(edim, edim)
        )

        # 第二次下采样 + 残差连接
        self.down2 = nn.Sequential(
            nn.Conv2d(edim, edim * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(edim * 2, edim * 2)
        )

        # 第三次下采样 + 残差连接
        self.down3 = nn.Sequential(
            nn.Conv2d(edim * 2, edim * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(edim * 4, out_channels)
        )

        self.tail_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, enc_x):
        x = self.down1(x)  # 第一次下采样
        x = self.down2(x)  # 第二次下采样
        x = self.down3(x)  # 第三次下采样
        x = self.tail_conv(torch.cat((x, enc_x), dim=1))
        return x


class ControlNet(nn.Module):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        dims, in_channels + hint_channels, model_channels, 3, padding=1
                    )
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        if 0:
            print("Ablation with DSM !! \n")
            self.DownInput = DownsampleModel(in_channels=3, out_channels=4)

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, channels, 2 * channels, 1, padding=0))
        )

    # def make_zero_conv(self, channels):
    #     return TimestepEmbedSequential(
    #         conv_2d_layers(channels, 2 * channels, 3, padding=1)
    #     )

    def forward(self, x_lq, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if 0:
            cons = self.DownInput(x_lq, hint)
        else:
            # print("ablation -----  w/o DSM \n")
            cons = hint

        x = torch.cat((x, cons), dim=1)
        outs = []

        h, emb, context = map(lambda t: t.type(self.dtype), (x, emb, context))
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)

        outs.append(self.middle_block_out(h, emb, context))

        return outs
