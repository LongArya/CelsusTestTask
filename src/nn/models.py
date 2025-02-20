import torch
from functools import partial
from torchsummary import summary
from torchvision.models import mobilenet_v3_small, MobileNetV3
from torchvision.ops.misc import Conv2dNormActivation
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
from typing import List
import torch.nn as nn
from ..consts import IMG_SIZE


bneck_conf = partial(InvertedResidualConfig, dilation=1, width_mult=1)
norm_layer = partial(nn.BatchNorm2d)


class MobileNetV3Backbone(nn.Module):
    def __init__(self):
        super(MobileNetV3Backbone, self).__init__()
        mobilenet = mobilenet_v3_small()
        self.backbone = mobilenet.features
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return x


class MobileNetV3LikeConvBackbone(nn.Module):
    """
    Mobile net backbone heavily inspired by MobileNetV3, adapted for small resolution
    Main changes:
        - reduced InvertedResidual block num
        - reduced downsampling rate


    Original small mobilenetV3 settings for reference
    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
    ]
    """

    def __init__(self, output_emb_size: int):
        super().__init__()
        # order: in_ch, ksize, expanded_channels, out_ch, use_se, act, stride
        inverted_residual_configs: List[InvertedResidualConfig] = [
            bneck_conf(16, 3, 16, 16, True, "RE", 1),
            bneck_conf(16, 3, 72, 24, False, "RE", 2),
            bneck_conf(24, 3, 88, 40, False, "RE", 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 2),
            bneck_conf(40, 5, 120, 48, True, "HS", 1),
            bneck_conf(48, 5, 144, output_emb_size, True, "HS", 1),
        ]

        start_conv = Conv2dNormActivation(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,  # No downsampling in the stem
            padding=1,
            norm_layer=nn.BatchNorm2d,
            activation_layer=nn.Hardswish,
        )

        # Build inverted residual blocks
        layers = [start_conv]
        for config in inverted_residual_configs:
            layers.append(InvertedResidual(config, norm_layer))

        self.features = nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return x


class Point2dRegressionHead(nn.Module):
    """Simple head, used for regressing object center in normalized coordinates"""

    def __init__(self, input_emb_size: int):
        super().__init__()
        self.regression_head = nn.Linear(in_features=input_emb_size, out_features=2)
        # we regress normalized coordinates between 0 and 1, so sigmoid is used
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regression_head(x)
        x = self.activation(x)
        return x


class CenterRegressionModel(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self._embedding_size = embedding_size
        self.backbone = MobileNetV3LikeConvBackbone(self._embedding_size)
        self.head = Point2dRegressionHead(self._embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x


class VanillaSiameseNetwork2dRepr(nn.Module):
    """Vanilla siamese netowork that used shared bbone for both inputs"""

    def __init__(self, embedding_size: int):
        super().__init__()
        self._embedding_size = embedding_size
        self.backbone = MobileNetV3LikeConvBackbone(self._embedding_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1 = self.backbone(x1)
        out2 = self.backbone(x2)
        return out1, out2


class VanillaSiameseNetwork2dRepr(nn.Module):
    """Vanilla siamese netowork that outputs 2d points"""

    def __init__(self, embedding_size: int):
        super().__init__()
        self._embedding_size = embedding_size
        self.backbone = MobileNetV3LikeConvBackbone(self._embedding_size)
        self.head = Point2dRegressionHead(self._embedding_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # print("INSIDE MODEL")
        # print(
        #     f"GOT 2 INPUTS x1={x1.size()}, x2={x2.size()}, close={torch.allclose(x1, x2, atol=1e-6)}"
        # )
        out1 = self.backbone(x1)
        out2 = self.backbone(x2)
        # print("====")
        # if self.training:
        #     print("TRAIN NORM = ", torch.linalg.norm(out1))
        # else:
        #     print("VAL NORM = ", torch.linalg.norm(out1))
        # print(f"CREATED 2 BBONE close={torch.allclose(out1, out2, atol=1e-4)}")
        out1 = self.head(out1)
        out2 = self.head(out2)
        # print(f"CREATED 2 BBONE close={torch.allclose(out1, out2, atol=1e-05)}")
        return out1, out2


class VanillaSiameseNetworkMobileNetV3Based(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV3Backbone()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out1 = self.backbone(x1)
        out2 = self.backbone(x2)
        return out1, out2


class SiameseNetworkWithRegressionHead(nn.Module):
    """
    Vanilla siamese netowork,
    but while in train mode, uses additional head
    for object center regression
    """

    def __init__(self, embedding_size: int):
        super().__init__()
        self._embedding_size = embedding_size
        self.backbone = MobileNetV3LikeConvBackbone(self._embedding_size)
        self.head = Point2dRegressionHead(self._embedding_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        emb1 = self.backbone(x1)
        emb2 = self.backbone(x2)
        if not self.training:
            return emb1, emb2
        else:
            regression1 = self.head(emb1)
            regression2 = self.head(emb2)
            return emb1, regression1, emb2, regression2


def inspect_mobile_net_v3_structure():
    model = mobilenet_v3_small()
    x = torch.zeros((1, 3, 224, 224))
    for block in model.features:
        init_size = x.size()
        x = block(x)
        print(type(block), f"{list(init_size)[1:]} -> {list(x.size())[1:]}")


def inspect_custom_net_v3_structure():
    model = MobileNetV3LikeConvBackbone(2)
    x = torch.zeros((1, 3, 32, 32))
    for block in model.features:
        init_size = x.size()
        x = block(x)
        print(type(block), f"{list(init_size)[1:]} -> {list(x.size())[1:]}")


def print_weight_norms(model: nn.Module):
    """
    Prints the L2 norm of the weights for every layer in the model.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and "weight" in name:  # Only print weights, not biases
            weight_norm = torch.norm(param.data, p=2).item()  # L2 norm
            print(f"Weight norm for layer {name}: {weight_norm:.6f}")


def _debug_mobile():
    def get_random_input(h, w) -> torch.Tensor:
        x = torch.zeros((1, 3, h, w), dtype=torch.float32)
        x += torch.rand_like(x) / 255
        return x

    backbone = MobileNetV3LikeConvBackbone(128)
    head = Point2dRegressionHead(128)
    default_bbone = MobileNetV3Backbone()

    # input = get_random_input(32, 32)
    # print(torch.linalg.norm(backbone(input)))
    print("SINGLE BRANCH")
    print(head(backbone(get_random_input(32, 32))))
    print(head(backbone(get_random_input(32, 32))))
    print(head(backbone(get_random_input(32, 32))))
    print(head(backbone(get_random_input(32, 32))))

    # print(default_bbone(get_random_input(224, 224)))

    model = VanillaSiameseNetwork2dRepr(128)
    # model.eval()
    print("INFER SIAMESE")
    print(model(get_random_input(32, 32), get_random_input(32, 32)))


# _debug_mobile()
