from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

import timm
import torchgeo.models
from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights
from location_encoder import get_positional_encoding, get_neural_network, LocationEncoder
from datamodules.s2geo_dataset import S2Geo

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, in_channels=3):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(in_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, in_channels: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

def create_hex_mask(k):
    if k % 2 == 0:
        raise ValueError("k must be odd to relate to a regular hexagon number.")
    
    mask = torch.ones((k, k), dtype=torch.float32)
    
    # Set upper right corner to zero
    for i in range((k - 1) // 2):
        mask[i, -(k - 1) // 2 + i:] = 0
    
    # Set lower left corner to zero
    for i in range((k - 1) // 2):
        mask[-(i + 1), :((k - 1) // 2 - i)] = 0
    
    return mask

class hexmaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, mask=None):
        super().__init__()

        # Check if kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd to relate to a regular hexagon number.")
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        if mask is None:
            self.mask = create_hex_mask(kernel_size)
        else:
            if mask.size() != torch.Size(kernel_size):
                raise ValueError(f"Mask size {mask.size()} does not match kernel size {torch.Size(kernel_size)}")
            self.mask = mask

    def forward(self, x):
        # Ensure the mask is on the same device as the input
        self.mask = self.mask.to(x.device)
        self.conv.weight = nn.Parameter(self.conv.weight * self.mask)
        return self.conv(x)
    
    @property
    def weight(self):
        return self.conv.weight
    
class Conv_on_hex(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, input_dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        # Check if kernel_size is either a single number or the same size as out_channels
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(out_channels)
        elif isinstance(kernel_size, tuple) or isinstance(kernel_size, list):
            if len(kernel_size) != len(out_channels):
                raise ValueError("kernel_size must be either a single number or the same size as out_channels")
        else:
            raise ValueError("kernel_size must be an int, tuple, or list")
        
        # Check if stride is either a single number or the same size as out_channels
        if isinstance(stride, int):
            stride = [stride] * len(out_channels)
        elif isinstance(stride, tuple) or isinstance(stride, list):
            if len(stride) != len(out_channels):
                raise ValueError("stride must be either a single number or the same size as out_channels")
        else:
            raise ValueError("stride must be an int, tuple, or list")
        
        # Check if padding is either a single number or the same size as out_channels
        if isinstance(padding, int):
            padding = [padding] * len(out_channels)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) != len(out_channels):
                raise ValueError("padding must be either a single number or the same size as out_channels")
        else:
            raise ValueError("padding must be an int, tuple, or list")
        
        self.layers = nn.ModuleList()
        
        for i, out_ch in enumerate(out_channels):
            in_ch = in_channels if i == 0 else out_channels[i-1]
            self.layers.append(nn.Sequential(
                hexmaskedConv2d(in_ch, out_ch, kernel_size[i], stride[i], padding[i]),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ))
        
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(out_channels[-1] * input_dim * input_dim, emb_dim)
        self.ln1d = nn.LayerNorm(emb_dim)
    
        self.dtype = self.layers[0][0].weight.dtype
    
    def get_dtype(self):
        return self.dtype

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.ln1d(x)
        return x

class ImageVec2Vec(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)
        self.ln1d = nn.LayerNorm(output_dim)
    
    @property
    def dtype(self):
        return self.fc.weight.dtype

    def forward(self, x):
        x = self.fc(x)
        x = self.ln1d(x)
        return x

class SatCLIP(nn.Module):
    def __init__(self, 
                 GS_dim: int, 
                 GS_model: int, 
                 GS_trainable: bool, 
                 OSM_dim: int, 
                 OSM_model: str, 
                 OSM_trainable: bool, 
                 hex_numb_rings: int, 
                 hex_in_channels: int, 
                 OSM_conv_layers: Tuple[int,...], 
                 S2_dim: int, 
                 S2_model: Union[Tuple[int, int, int, int], int, str], 
                 S2_trainable: bool, 
                 vision_width: int, 
                 vision_patch_size: int,
                 FM_dim: int,
                 FM_model: str, 
                 FM_trainable: bool, 
                 FM_conv_layers: Tuple[int,...], 
                 Combined_dim: int, 
                 Combined_layers: int, 
                 Combined_capacity: int, 
                 pos_encoder: str, 
                 loc_encoder: str, 
                 loc_layers: int, 
                 loc_capacity: int, 
                 image_resolution: int,
                 frequency_num: int, 
                 max_radius: int, 
                 min_radius: int,
                 harmonics_calculation: str, 
                 legendre_polys: int, 
                 sh_embedding_dims: int, 
                 S2_channels: int=12,
                 *args, 
                 **kwargs):
        super().__init__()

        # ---------------------------------- #
        # GS view

        if GS_model == 'moco_resnet18':
            print(f'GS_dim is {GS_dim}, using pretrained moco ResNet18 for GS images, model trainable is {GS_trainable}')
            weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
            in_chans = weights.meta["in_chans"]
            self.GSview = timm.create_model("resnet18", in_chans=in_chans, num_classes=GS_dim)
            self.GSview.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.GSview.requires_grad_(GS_trainable)
            self.GSview.fc.requires_grad_(True)

        elif GS_model == 'vectorized':
            print(f'GS_dim is {GS_dim}, using forward layer on vectored GS images')
            self.GSview = ImageVec2Vec(input_dim=512, output_dim=GS_dim)

        elif GS_model == 'moco_resnet50':
            print(f'GS_dim is {GS_dim}, using pretrained moco ResNet50 for GS images, model trainable is {GS_trainable}')
            weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
            in_chans = weights.meta["in_chans"]
            self.GSview = timm.create_model("resnet50", in_chans=in_chans, num_classes=GS_dim)
            self.GSview.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.GSview.requires_grad_(GS_trainable)
            self.GSview.fc.requires_grad_(True)

        else:
            print('no GS model requested')
            self.GSview = None

        # ---------------------------------- #
        # OSM view

        if OSM_model == 'hexconv':
            print(f'OSM_dim is {OSM_dim}, using hexagonal convolutional embedding for OSM data')
            self.OSMview = Conv_on_hex(in_channels=hex_in_channels, 
                           out_channels=OSM_conv_layers, 
                           emb_dim=OSM_dim, 
                           input_dim=2*hex_numb_rings+1, 
                           kernel_size=3, 
                           stride=1, 
                           padding=1)
        else:
            print('No OSM model requested')
            self.OSMview = None
        
        # ---------------------------------- #
        # S2 view
        
        if isinstance(S2_model, (tuple, list)):
            print(f'S2_dim is {S2_dim}, using modified resnet for S2 images')
            vision_heads = vision_width * 32 // 64
            self.S2view = ModifiedResNet(
                layers=S2_model,
                output_dim=S2_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                in_channels=S2_channels
            )
        elif S2_model == 'moco_resnet18':
            print(f'S2_dim is {S2_dim}, using pretrained moco resnet18 for S2 images, model trainable is {S2_trainable}')
            weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.S2view = timm.create_model("resnet18", in_chans=in_chans, num_classes=S2_dim)
            self.S2view.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.S2view.requires_grad_(S2_trainable)
            self.S2view.fc.requires_grad_(True)
        elif S2_model == 'moco_resnet50':
            print(f'S2_dim is {S2_dim}, using pretrained moco resnet50 for S2 images, model trainable is {S2_trainable}')
            weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.S2view = timm.create_model("resnet50", in_chans=in_chans, num_classes=S2_dim)
            self.S2view.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.S2view.requires_grad_(S2_trainable)
            self.S2view.fc.requires_grad_(True)
        elif S2_model == 'moco_vit16':
            print(f'S2_dim is {S2_dim}, using pretrained moco vit16 for S2 images, model trainable is {S2_trainable}')
            weights = ViTSmall16_Weights.SENTINEL2_ALL_MOCO
            in_chans = weights.meta["in_chans"]
            self.S2view = timm.create_model("vit_small_patch16_224", in_chans=in_chans, num_classes=S2_dim)
            self.S2view.load_state_dict(weights.get_state_dict(progress=True), strict=False)
            self.S2view.requires_grad_(S2_trainable)
            self.S2view.head.requires_grad_(True)
        elif S2_model == 'transformer':
            print(f'S2_dim is {S2_dim}, using vision transformer for S2 images')
            self.S2view = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=S2_model,
            heads=vision_heads,
            output_dim=S2_dim,
            in_channels=S2_channels
            )
        else:
            print('No S2 model requested')
            self.S2view = None

        # ---------------------------------- #
        # FM view
        # TODO: Implement FM view

        print('No floodmap model requested')
        self.FMview = None
        
        # ---------------------------------- #                       
        # Combined view

        if Combined_layers == 'None':
            print('No combined layers requested')
            self.combined_view = lambda x: x  # Identity function, no trainable parameters
        elif Combined_layers == 0:
            print(f'Combining views from dimension GS_dim={GS_dim}, OSM_dim={OSM_dim}, S2_dim={S2_dim}, FM_dim={FM_dim} to Combined_dim={Combined_dim}')
            self.combined_view = nn.Linear(GS_dim + OSM_dim + S2_dim + FM_dim, Combined_dim)
        else:
            print(f'Combining views from dimension GS_dim={GS_dim}, OSM_dim={OSM_dim}, S2_dim={S2_dim}, FM_dim={FM_dim} to Combined_dim={Combined_dim} via {Combined_layers} layers of size {Combined_capacity}')
            combined_layers = []
            combined_layers.append(nn.Linear(GS_dim + OSM_dim + S2_dim + FM_dim, Combined_capacity))
            combined_layers.append(nn.ReLU(inplace=True))
            for _ in range(Combined_layers - 1):
                combined_layers.append(nn.Linear(Combined_capacity, Combined_capacity))
                combined_layers.append(nn.ReLU(inplace=True))
            combined_layers.append(nn.Linear(Combined_capacity, Combined_dim))
            combined_layers.append(nn.LayerNorm(Combined_dim))
            
            self.combined_view = nn.Sequential(*combined_layers)

        # ---------------------------------- #
        # Location view	
        
        self.posenc = get_positional_encoding(name=pos_encoder, harmonics_calculation=harmonics_calculation, legendre_polys=legendre_polys, min_radius=min_radius, max_radius=max_radius, frequency_num=frequency_num).double()
        self.nnet = get_neural_network(name=loc_encoder, input_dim=self.posenc.embedding_dim, num_classes=Combined_dim, dim_hidden=loc_capacity, num_layers=loc_layers).double()
        self.location = LocationEncoder(self.posenc, 
                                        self.nnet
        ).double()
        
        # ---------------------------------- #
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.S2view, ModifiedResNet):
            if self.S2view.attnpool is not None:
                std = self.S2view.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.S2view.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.S2view.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.S2view.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.S2view.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.S2view.layer1, self.S2view.layer2, self.S2view.layer3, self.S2view.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        if isinstance(self.GSview, timm.models.vision_transformer.VisionTransformer):
            return self.GSview.patch_embed.proj.weight.dtype
        elif hasattr(self.GSview, 'conv1'):
            return self.GSview.conv1.weight.dtype
        elif hasattr(self.GSview, 'dtype'):
            return self.GSview.dtype
        elif hasattr(self.S2view, 'conv1'):
            return self.S2view.conv1.weight.dtype
        elif isinstance(self.S2view, timm.models.vision_transformer.VisionTransformer):
            return self.S2view.patch_embed.proj.weight.dtype
        elif hasattr(self.S2view, 'dtype'):
            return self.S2view.dtype
        elif hasattr(self.FMview, 'dtype'):
            return self.FMview.dtype
        else:
            raise ValueError("Cannot determine dtype from GSview, OSMview, S2view, or FMview")

    def encode_GS(self, gs):
        if self.GSview is not None:
            return self.GSview(gs.type(self.dtype))
        else:
            return torch.empty(1, 0)

    def encode_OSM(self, osm):
        if self.OSMview is not None:
            return self.OSMview(osm.to(self.OSMview.get_dtype()))
        else:
            return torch.empty(1, 0)
        
    def encode_s2(self, s2):
        if self.S2view is not None:
            return self.S2view(s2.type(self.dtype))
        else:
            return torch.empty(1, 0)
        
    def encode_fm(self, fm):
        if self.FMview is not None:
            return self.FMview(fm.type(self.dtype))
        else:
            return torch.empty(1, 0)
       
    def encode_combined(self, comb):
        if self.combined_view is not None:
            return self.combined_view(comb)
        else:
            return None
        
    def encode_location(self, coords):
        return self.location(coords.double())

    def forward(self, gs, osm, s2, fm, coords):
        features = []

        if self.GSview is not None:
            features.append(self.encode_GS(gs))
        if self.OSMview is not None:
            features.append(self.encode_OSM(osm))
        if self.S2view is not None:
            features.append(self.encode_s2(s2))
        if self.FMview is not None:
            features.append(self.encode_fm(fm))

        combined_features = self.encode_combined(torch.cat(features, dim=1))

        location_features = self.encode_location(coords).float()
        
        # Normalize features
        combined_features = combined_features / combined_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_input = logit_scale * combined_features @ location_features.t()
        logits_per_location = logits_per_input.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_input, logits_per_location

def load_checkpoint(model, checkpoint_path):
    """
    Load weights from a checkpoint file into the model.

    Parameters:
    - model: The model instance to load the weights into.
    - checkpoint_path: Path to the checkpoint file.
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Assuming the checkpoint contains state_dicts for GSview, OSMview, S2view, FMview, Combinedview, and location models
    gs_state_dict = checkpoint['GSview']
    osm_state_dict = checkpoint['OSMview']
    s2_state_dict = checkpoint['S2view']
    fm_state_dict = checkpoint['FMview']
    combined_state_dict = checkpoint['combined_view']
    location_state_dict = checkpoint['location']
    
    # Load the state dictionaries into the respective models
    model.GSview.load_state_dict(gs_state_dict)
    model.OSMview.load_state_dict(osm_state_dict)
    model.S2view.load_state_dict(s2_state_dict)
    model.FMview.load_state_dict(fm_state_dict)
    model.Combinedview.load_state_dict(combined_state_dict)
    model.location.load_state_dict(location_state_dict)

    print("Checkpoint loaded successfully.")

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)