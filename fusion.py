# ------------------------------------------------------------
# Author: Viet Hoang Le (Mikyx-1)
# Date: December 8th, 2024
# Description: 
#     This script fuses BatchNorm layers into Conv layers for 
#     deep learning models in PyTorch, optimizing them for inference.
# GitHub: https://github.com/Mikyx-1 (if applicable)
# License: MIT License (if applicable)
# ------------------------------------------------------------


import torch
from torch import nn
import functools
from copy import deepcopy
from tqdm import tqdm
import inspect
import time
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

def extract_layers_hierarchy(module, prefix="", depth=0):
    """
    Recursively extract layers from a PyTorch module, grouping Conv2d and BatchNorm2d layers.

    Args:
        module (nn.Module): The PyTorch model or submodule.
        prefix (str): Prefix for naming layers to include parent module hierarchy.
        depth (int): Current depth of recursion, used for debugging or hierarchy tracking.

    Returns:
        list: List of grouped Conv2d and BatchNorm2d layers in order.
    """
    layers = []
    prev_conv = None

    for name, submodule in module.named_children():
        full_name = f"{prefix}{name}" if prefix else name

        if len(list(submodule.children())) > 0:  # Check if the submodule has children
            # Recursively extract from children modules
            layers.extend(extract_layers_hierarchy(submodule, prefix=f"{full_name}.", depth=depth + 1))
        else:
            # Check if the current layer is Conv2d or BatchNorm2d
            layer_type = type(submodule).__name__
            if layer_type == "Conv2d":
                prev_conv = full_name  # Store the Conv2d layer temporarily
            elif layer_type == "BatchNorm2d" and prev_conv:
                # If a BatchNorm2d follows a Conv2d, group them
                layers.append([prev_conv, full_name])
                prev_conv = None
            else:
                # Reset if a Conv2d is not followed by BatchNorm2d
                prev_conv = None

    return layers

def check_nested_hasattr(obj: object, attr_path: str):
    try:
        attrs = attr_path.split(".")
        for attr in attrs:
            obj = getattr(obj, attr)
        return True
    except AttributeError:
        return False
    
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
    print(f"{Fore.GREEN}Phase 1 started: Fusing BatchNorm layers into Conv layers.{Style.RESET_ALL}")
    
    fused_model = deepcopy(model)
    fuseable_layer_attributes = extract_layers_hierarchy(model)

    params_reduced = 0

    for fuseable_layer_attribute in tqdm(fuseable_layer_attributes, desc="Fusing Layers"):
        conv_layer = get_layer_by_path(fused_model, fuseable_layer_attribute[0])
        bn_layer = get_layer_by_path(fused_model, fuseable_layer_attribute[1])
        if isinstance(bn_layer, nn.Identity):
            continue
        # Fuse conv and bn layers
        fused_layer = fuse_conv_and_bn(conv_layer, bn_layer)
        num_conv_params = sum(p.numel() for p in conv_layer.parameters())
        num_bn_params = sum(p.numel() for p in bn_layer.parameters())
        num_fused_params = sum(p.numel() for p in fused_layer.parameters())
        params_reduced += num_conv_params + num_bn_params - num_fused_params
        rsetattr(fused_model, fuseable_layer_attribute[0], fused_layer)
        rsetattr(fused_model, fuseable_layer_attribute[1], nn.Identity())

    print(f"Phase 1 completed: BatchNorm fusion finished. {params_reduced} parameters were reduced after fusion.")

    # Phase 2: Similarity test (colored in blue)
    print(f"{Fore.BLUE}Phase 2 started: Checking the fused model.{Style.RESET_ALL}")
    run_similarity_test_with_progress(model, fused_model)  # Assuming this function provides its own progress updates
    print(f"Phase 2 completed: Similarity test completed.")

    return fused_model

def infer_input_from_model(model: nn.Module):
    """
    Infer the expected input shapes of a model, including models with multiple inputs.

    Args:
        model (nn.Module): The model whose input shape needs to be inferred.

    Returns:
        tuple or torch.Tensor: A tuple of input tensors for multiple inputs,
        or a single tensor if the model has only one input.
    """
    # Inspect the forward method signature to find argument names and count
    forward_sig = inspect.signature(model.forward)
    param_names = list(forward_sig.parameters.keys())

    # Exclude 'self' and 'kwargs'-like parameters
    param_names = [
        name for name in param_names
        if name != 'self' and forward_sig.parameters[name].kind != inspect.Parameter.VAR_KEYWORD
    ]

    if not param_names:
        raise ValueError("Unable to infer input: no valid parameters found in the forward method.")

    # Assume the default shape for each input is (1, 3, 224, 224)
    # This can be adjusted based on your domain knowledge
    example_inputs = []
    for param in param_names:
        default_shape = (1, 3, 224, 224)  # Assumes image input for each tensor
        example_inputs.append(torch.randn(*default_shape))

    # Convert to a tuple if multiple inputs, otherwise return a single tensor
    return tuple(example_inputs) if len(example_inputs) > 1 else example_inputs[0]


@torch.no_grad()
def run_similarity_test_with_progress(model, fused_model, iterations: int = 10):
    """
    Run similarity tests between the original and fused models with a progress bar
    and measure inference speed.

    Args:
        model (nn.Module): Original model.
        fused_model (nn.Module): Fused model.
        iterations (int): Number of iterations for testing.
    """
    model.eval()
    fused_model.eval()
    print("Inferring input from the model...")
    input_sample = infer_input_from_model(model)
    print(f"Inferred input: {[tuple(inp.shape) for inp in input_sample] if isinstance(input_sample, tuple) else input_sample.shape}")

    # Progress bar for iterations
    tol = 1e-2
    differences = []
    model_times = []
    fused_model_times = []

    with tqdm(total=iterations, desc="Running similarity test") as pbar:
        for _ in range(iterations):
            # Generate random input based on inferred shapes
            if isinstance(input_sample, tuple):
                sample = tuple(torch.randn_like(inp) for inp in input_sample)
            else:
                sample = torch.randn_like(input_sample)

            # Measure original model inference time
            start_time = time.time()
            if isinstance(input_sample, tuple):
                model_res = model(*sample)  # Unpack for multiple inputs
            else:
                model_res = model(sample)
            model_times.append(time.time() - start_time)

            # Measure fused model inference time
            start_time = time.time()
            if isinstance(input_sample, tuple):
                fused_res = fused_model(*sample)
            else:
                fused_res = fused_model(sample)
            fused_model_times.append(time.time() - start_time)

            # Compute difference
            difference = torch.linalg.norm(model_res - fused_res)
            differences.append(difference.item())
            pbar.update(1)

    # Compute average inference times
    avg_model_time = sum(model_times) / iterations
    avg_fused_model_time = sum(fused_model_times) / iterations
    speedup = (avg_model_time - avg_fused_model_time) / avg_model_time * 100

    # Report results
    max_diff = max(differences)
    print(f"\nMaximum difference: {max_diff}")
    if max_diff > tol:
        print(f"WARNING: Difference exceeds tolerance of {tol}")
    else:
        print("Models are within tolerance!")

    # Report speedup
    print(f"Average inference time (original model): {avg_model_time:.6f} seconds")
    print(f"Average inference time (fused model): {avg_fused_model_time:.6f} seconds")
    print(f"Fused model is {speedup:.2f}% faster than the original model.")