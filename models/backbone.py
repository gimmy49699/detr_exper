# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

# 按需求新增backbone


import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from models.position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


# DETR原始backbone
class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # pretrained=is_main_process() => 如果时main进程，则自动从官方载入预训练模型
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


# 修改后的Backbone
class MyBackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool, mode: str):
        super().__init__()
        # for sub_backbone in backbone:
        #     print(sub_backbone)
        #     for name, parameter in sub_backbone.named_parameters():
        #         if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #             print(name, "  is not updated!")
        #             parameter.requires_grad_(False)

        # 修改的Backbone不支持多层输出
        return_layers = {'layer4': "0"}
        self.body = nn.ModuleList([
            backbone[0],
            IntermediateLayerGetter(backbone[1], return_layers=return_layers)
        ])
        self.projector = nn.Linear(2 * num_channels, num_channels)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = []
        for body in self.body:
            # IntermediateLayerGetter 可以输出多层，这里仅输出resnet最后一层
            xs.append(body(tensor_list.tensors)["0"])
        xs = torch.concat(xs, dim=1)
        N, C, W, H = xs.size()
        xs = xs.reshape(N, W * H, C)

        xs = self.projector(xs)
        xs = xs.reshape(N, C // 2, W, H)
        _xs = OrderedDict()
        _xs["0"] = xs

        out: Dict[str, NestedTensor] = {}
        for name, x in _xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class ConvDilateNet(nn.Module):
    """Dilated-CNN for backbone."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        )

        block_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        )

        block_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, dilation=4),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        )

        self.conv_blocks = nn.ModuleList([block_1, block_2, block_3])
        self.project_layer = nn.Linear(512 * 3, out_channels)

    def forward(self, x):
        N, C, W, H = x.size()
        xs = []
        for block in self.conv_blocks:
            _xs = block(x)
            _, _, _w, _h = _xs.size()
            _xs = F.pad(_xs, [0, (W - _w), 0, (H - _h)], value=0)
            xs.append(_xs)
        xs = torch.concat(xs, dim=1)

        n, c, w, h = xs.size()
        xs = self.project_layer(xs.reshape(n, w * h, c)).reshape(N, C, W, H) + x
        return xs


class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size)

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class ResnetWithDilateConv(nn.Module):

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        super().__init__()

        return_layers = {'layer4': "0"}
        resnet = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=False, norm_layer=FrozenBatchNorm2d)
        self.resnet = IntermediateLayerGetter(resnet, return_layers=return_layers)
        self.conv = ConvDilateNet(in_channels=2048)

    def forward(self, x):
        out = self.resnet(x)
        out = self.conv(out["0"])
        output = OrderedDict()
        output["0"] = out
        return output


class MyBackbone(MyBackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 mode="1"):
        # pretrained=is_main_process() => 如果时main进程，则自动从官方载入预训练模型
        backbone_left = ResnetWithDilateConv(name, train_backbone, return_interm_layers, dilation)

        if mode == "1":
            backbone_right = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=False, norm_layer=FrozenBatchNorm2d)

        if mode == "1":
            backbone = nn.ModuleList([backbone_left, backbone_right])
        elif mode == "2":
            backbone = nn.ModuleList([backbone_left])
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, mode)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.new_backbone:
        backbone = MyBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

#  modifications
# x = NestedTensor(torch.rand(2, 3, 200, 250), torch.rand(3, 200, 250))


# model = Resnet_with_DilateConv(name='resnet50',
#     train_backbone=True,
#     return_interm_layers=False,
#     dilation=False)
#
# model = IntermediateLayerGetter(model, return_layers={'layer4':'0'})
# model = MyBackbone(
#     name='resnet50',
#     train_backbone=True,
#     return_interm_layers=False,
#     dilation=False
# )
# output = model(x)
# out, mask = output['0'].decompose()
# print(out.shape, mask.shape)
# print(model)
