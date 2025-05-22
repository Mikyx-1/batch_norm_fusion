# ------------------------------------------------------------
# Author: Viet Hoang Le (Mikyx-1)
# Date: December 8th, 2024
# Description:
# This script fuses BatchNorm layers into Conv layers for
# deep learning models in PyTorch, optimising them for inference.
# GitHub: https://github.com/Mikyx-1/batch_norm_fusion
# License: MIT License
# ------------------------------------------------------------
import functools
from copy import deepcopy

import torch
import torch.nn as nn
from colorama import Fore, Style
from torch import nn
from tqdm import tqdm

from utils import fuse_conv_bn_eval

torch.manual_seed(42)


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
    # Ensure both layers are in eval mode for fusion
    conv.eval()
    bn.eval()

    with torch.no_grad():
        # Use the reference implementation for all cases
        fused_layer = fuse_conv_bn_eval(conv, bn)
        return fused_layer


def fuse_conv_and_bn_manual(conv, bn):
    """
    Manual implementation of Conv2d + BatchNorm2d fusion.
    This is a corrected version of the original manual implementation.

    Args:
        conv (torch.nn.Conv2d): The convolutional layer.
        bn (torch.nn.BatchNorm2d): The batch normalization layer.

    Returns:
        torch.nn.Conv2d: The fused convolutional layer.
    """
    # Ensure both layers are in eval mode
    conv.eval()
    bn.eval()

    with torch.no_grad():
        # Create a new Conv2d layer with the same parameters, ensuring bias=True
        fusedconv = nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).to(device=conv.weight.device, dtype=conv.weight.dtype)

        # Get BN parameters
        bn_eps = bn.eps
        bn_var = bn.running_var
        bn_mean = bn.running_mean
        bn_weight = bn.weight if bn.weight is not None else torch.ones_like(bn_mean)
        bn_bias = bn.bias if bn.bias is not None else torch.zeros_like(bn_mean)

        # Compute the scale factor: bn_weight / sqrt(bn_var + bn_eps)
        bn_var_rsqrt = torch.rsqrt(bn_var + bn_eps)
        scale_factor = bn_weight * bn_var_rsqrt

        # Get convolution weights and bias
        conv_weight = conv.weight
        conv_bias = (
            conv.bias
            if conv.bias is not None
            else torch.zeros(
                conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype
            )
        )

        # Fuse weights: conv_weight * scale_factor
        # Scale factor needs to be reshaped to broadcast correctly
        # For conv weights with shape [out_channels, in_channels, H, W]
        # Scale factor shape should be [out_channels, 1, 1, 1]
        scale_shape = [conv.out_channels] + [1] * (len(conv_weight.shape) - 1)
        fused_weight = conv_weight * scale_factor.view(scale_shape)

        # Fuse bias: (conv_bias - bn_mean) * scale_factor + bn_bias
        fused_bias = (conv_bias - bn_mean) * scale_factor + bn_bias

        # Copy the fused parameters
        fusedconv.weight.copy_(fused_weight)
        fusedconv.bias.copy_(fused_bias)

        return fusedconv


def find_conv_bn_pairs(traced_model):
    """
    Identifies Conv2d-BatchNorm2d pairs in a traced model.

    Args:
        traced_model: The symbolically traced PyTorch model.

    Returns:
        list: List of tuples (conv_name, bn_name) for Conv-BN pairs.
    """
    conv_bn_pairs = []
    prev_node = None
    module_dict = dict(traced_model.named_modules())

    for node in traced_model.graph.nodes:
        if node.op == "call_module":
            module = module_dict[node.target]
            if isinstance(module, nn.Conv2d):
                prev_node = node
            elif isinstance(module, nn.BatchNorm2d) and prev_node:
                conv_name = prev_node.target
                bn_name = node.target
                conv_bn_pairs.append((conv_name, bn_name))
                prev_node = None
    return conv_bn_pairs


def get_layer_by_path(model, layer_path):
    """
    Retrieves a layer from a model using a dot-separated path.

    Args:
        model (nn.Module): The model.
        layer_path (str): Dot-separated path to the layer.

    Returns:
        nn.Module: The requested layer.
    """
    attrs = layer_path.split(".")
    layer = model
    for attr in attrs:
        if attr.isdigit():
            layer = layer[int(attr)]
        else:
            layer = getattr(layer, attr)
    return layer


def rsetattr(obj, attr, val):
    """
    Sets an attribute on an object using a dot-separated path.

    Args:
        obj: The object.
        attr (str): Dot-separated attribute path.
        val: Value to set.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
    Gets an attribute from an object using a dot-separated path.

    Args:
        obj: The object.
        attr (str): Dot-separated attribute path.

    Returns:
        The requested attribute.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


@torch.no_grad()
def fuse(model):
    """
    Fuses BatchNorm layers into Conv layers for a given model.

    Args:
        model (nn.Module): The original model to be fused.

    Returns:
        nn.Module: A fused model with BatchNorm layers merged into Conv layers.
    """
    model.eval()
    device = next(model.parameters()).device
    for _ in range(10):
        model(torch.randn(1, 3, 224, 224, device=device))

    print(
        f"{Fore.GREEN}Process started: Fusing BatchNorm layers into Conv layers.{Style.RESET_ALL}"
    )
    fused_model = deepcopy(model)
    fused_model.eval()
    traced = torch.fx.symbolic_trace(fused_model)
    fuseable_layer_attributes = find_conv_bn_pairs(traced)
    params_reduced = 0

    for conv_name, bn_name in tqdm(fuseable_layer_attributes, desc="Fusing Layers"):
        try:
            conv_layer = get_layer_by_path(fused_model, conv_name)
            bn_layer = get_layer_by_path(fused_model, bn_name)
            if isinstance(bn_layer, nn.Identity):
                continue

            test_input = torch.randn(
                1, conv_layer.in_channels, 32, 32, device=conv_layer.weight.device
            )
            original_output = bn_layer(conv_layer(test_input))
            fused_layer = fuse_conv_and_bn(conv_layer, bn_layer)
            fused_output = fused_layer(test_input)
            diff = torch.linalg.norm(original_output - fused_output)
            if diff > 1e-6:  # Stricter threshold for precision
                print(
                    f"{Fore.YELLOW}Warning: Difference ({diff:.6f}) for {conv_name} and {bn_name}{Style.RESET_ALL}"
                )

            num_conv_params = sum(p.numel() for p in conv_layer.parameters())
            num_bn_params = sum(p.numel() for p in bn_layer.parameters())
            num_fused_params = sum(p.numel() for p in fused_layer.parameters())
            params_reduced += num_conv_params + num_bn_params - num_fused_params

            rsetattr(fused_model, conv_name, fused_layer)
            rsetattr(fused_model, bn_name, nn.Identity())
        except Exception as e:
            print(
                f"{Fore.YELLOW}Error fusing {conv_name} and {bn_name}: {e}. Skipping.{Style.RESET_ALL}"
            )
            continue

    print(
        f"Fusion completed: {Fore.GREEN}{params_reduced} parameters reduced after fusion.{Style.RESET_ALL}"
    )
    return fused_model


# Example usage
if __name__ == "__main__":
    import torchvision.models

    model = torchvision.models.resnet152(weights=None)
    model.eval()
    fused_model = fuse(model)
    fused_model.eval()

    dummy = torch.randn((1, 3, 224, 224))
    res1 = model(dummy)
    res2 = fused_model(dummy)
    output_difference = torch.linalg.norm(res1 - res2)
    relative_difference = output_difference / torch.linalg.norm(res1)
    print(f"Output difference after fusion: {output_difference}")
    print(f"Relative output difference after fusion: {relative_difference}")
