# ------------------------------------------------------------
# Author: Viet Hoang Le (Mikyx-1)
# Date: December 8th, 2024
# Description: 
#     This script fuses BatchNorm layers into Conv layers for 
#     deep learning models in PyTorch, optimising them for inference.
# GitHub: https://github.com/Mikyx-1/batch_norm_fusion
# License: MIT License
# ------------------------------------------------------------


import torch
from torch import nn
import functools
from copy import deepcopy
from tqdm import tqdm
from colorama import Fore, Style

def fuse_conv_and_bn(conv, bn):
    """
    Fuses a Conv2d layer and a BatchNorm2d layer into a single Conv2d layer.
    Supports both regular and grouped convolutions.

    Args:
        conv (torch.nn.Conv2d): The convolutional layer.
        bn (torch.nn.BatchNorm2d): The batch normalization layer.

    Returns:
        torch.nn.Conv2d: The fused convolutional layer.
    """
    # Initialize the fused convolution layer
    fusedconv = torch.nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,  # Preserve groups for grouped convolutions
        bias=True  # Always use bias in the fused layer
    )

    # Prepare convolution weights
    w_conv = conv.weight.clone()
    if conv.groups > 1:
        # For grouped convolutions, process weights group-wise
        w_conv = w_conv.view(conv.groups, -1, w_conv.size(1), w_conv.size(2), w_conv.size(3))  # Grouped view
        w_conv = w_conv.reshape(conv.out_channels, -1)  # Flatten across groups
    else:
        # For regular convolutions, flatten weights directly
        w_conv = w_conv.view(conv.out_channels, -1)

    # Compute the BN scaling factors
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))

    # Apply the scaling factor to the convolution weights
    fused_weight = torch.mm(w_bn, w_conv)
    fused_weight = fused_weight.view(fusedconv.weight.size())  # Reshape to original Conv2d weight shape
    fusedconv.weight.copy_(fused_weight)

    # Prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias.clone()
    else:
        b_conv = torch.zeros(conv.out_channels, device=conv.weight.device)

    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.matmul(w_bn, b_conv.unsqueeze(1)).squeeze() + b_bn)

    return fusedconv


def find_conv_bn_pairs(traced_model):
    conv_bn_pairs = []
    prev_node = None
    module_dict = dict(traced_model.named_modules())  # Get all modules with proper dot-separated names

    for node in traced_model.graph.nodes:
        if node.op == 'call_module':
            module = module_dict[node.target]
            if isinstance(module, nn.Conv2d):
                prev_node = node
            elif isinstance(module, nn.BatchNorm2d) and prev_node:
                # Use the full dot-separated module names
                conv_name = node.target  # Already in dot notation
                bn_name = prev_node.target  # Already in dot notation
                conv_bn_pairs.append((bn_name, conv_name))  # Keep order (conv, bn)
                prev_node = None
    return conv_bn_pairs

    
def get_layer_by_path(model, layer_path):
    attrs = layer_path.split(".")
    layer = model
    for attr in attrs:
        if attr.isdigit():
            layer = layer[int(attr)]
        else:
            layer = getattr(layer, attr)
    return layer

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


@torch.no_grad()
def fuse(model):
    """
    Fuses BatchNorm layers into Conv layers for a given model.
    
    Args:
        model (nn.Module): The original model to be fused.
    
    Returns:
        nn.Module: A fused model with BatchNorm layers merged into Conv layers.
    """
    # Phase 1: Fusion (colored in green)
    print(f"{Fore.GREEN}Process started: Fusing BatchNorm layers into Conv layers.{Style.RESET_ALL}")
    
    fused_model = deepcopy(model)
    fused_model.eval()
    traced = torch.fx.symbolic_trace(fused_model) 
    fuseable_layer_attributes = find_conv_bn_pairs(traced)  # Get conv-bn pairs
    params_reduced = 0

    for conv_name, bn_name in tqdm(fuseable_layer_attributes, desc="Fusing Layers"):
        try:
            conv_layer = get_layer_by_path(fused_model, conv_name)
            bn_layer = get_layer_by_path(fused_model, bn_name)
            
            if isinstance(bn_layer, nn.Identity):
                continue  # Already fused

            # Fuse conv and bn layers
            fused_layer = fuse_conv_and_bn(conv_layer, bn_layer)

            # Count parameter reduction
            num_conv_params = sum(p.numel() for p in conv_layer.parameters())
            num_bn_params = sum(p.numel() for p in bn_layer.parameters())
            num_fused_params = sum(p.numel() for p in fused_layer.parameters())
            params_reduced += num_conv_params + num_bn_params - num_fused_params

            # Replace layers
            rsetattr(fused_model, conv_name, fused_layer)
            rsetattr(fused_model, bn_name, nn.Identity())  # Remove BN after fusion
            
        except Exception as e:
            print(f"{Fore.YELLOW}Error fusing {conv_name} and {bn_name}: {e}. The code will skip fusing these layers.{Style.RESET_ALL}")
            continue

    print(f"Fusion completed: {Fore.GREEN}{params_reduced} parameters reduced after fusion.{Style.RESET_ALL}")

    return fused_model