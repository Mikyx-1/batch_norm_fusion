from typing import Dict, List, Union

import torch
import torch.nn as nn

torch.set_grad_enabled(False)


def extract_batch_norm_inputs(
    model: torch.nn.Module,
) -> Dict[str, Union[str, List[str]]]:
    """
    Extracts batch normalization layers and their input nodes from a PyTorch model's FX graph.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary mapping batch norm layer names to their input node names.
        If the input is from a cat operation, the value is a list of input node names.
    """
    # Symbolically trace the model
    traced = torch.fx.symbolic_trace(model)
    graph = traced.graph

    # Dictionary to store batch norm layer -> input(s) mapping
    bn_inputs = {}

    # Iterate through all nodes in the graph
    for node in graph.nodes:
        # Check if the node is a call_module and its target is a batch norm layer
        if node.op == "call_module":
            module = traced
            for part in node.target.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            # Check if the module is a batch norm layer
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Get the input node(s)
                input_nodes = node.args

                if len(input_nodes) == 1:
                    input_node = input_nodes[0]

                    # Check if the input node is a cat operation
                    if (
                        input_node.op == "call_function"
                        and "cat" in str(input_node.target).lower()
                    ):
                        # Get all inputs to the cat operation
                        cat_inputs = input_node.args[
                            0
                        ]  # First arg is the list of tensors
                        # Store the list of input node names
                        bn_inputs[node.target] = [str(inp) for inp in cat_inputs]
                    else:
                        # Single input case
                        bn_inputs[node.target] = str(input_node)
                else:
                    # Handle unexpected cases with multiple inputs
                    bn_inputs[node.target] = [str(inp) for inp in input_nodes]

    return bn_inputs


def extract_linear_batchnorm_inputs(
    model: torch.nn.Module,
) -> Dict[str, Union[str, List[str]]]:
    """
    Extracts Linear-BatchNorm pairs from a PyTorch model's FX graph.

    Args:
        model: PyTorch model to analyze

    Returns:
        Dictionary mapping batch norm layer names to their input node names.
        If the input is from a cat operation, the value is a list of input node names.
    """
    # Symbolically trace the model
    traced = torch.fx.symbolic_trace(model)
    graph = traced.graph

    # Dictionary to store batch norm layer -> input(s) mapping
    linear_bn_inputs = {}

    # Iterate through all nodes in the graph
    for node in graph.nodes:
        # Check if the node is a call_module and its target is a batch norm layer
        if node.op == "call_module":
            module = traced
            for part in node.target.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            # Check if the module is a batch norm layer
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Get the input node(s)
                input_nodes = node.args

                if len(input_nodes) == 1:
                    input_node = input_nodes[0]

                    # Check if the input node is a Linear layer
                    if input_node.op == "call_module":
                        input_module = traced
                        for part in input_node.target.split("."):
                            input_module = getattr(input_module, part, None)
                            if input_module is None:
                                break
                        if isinstance(input_module, nn.Linear):
                            linear_bn_inputs[node.target] = str(input_node)
                    # Check if the input node is a cat operation
                    elif (
                        input_node.op == "call_function"
                        and "cat" in str(input_node.target).lower()
                    ):
                        # Get all inputs to the cat operation
                        cat_inputs = input_node.args[
                            0
                        ]  # First arg is the list of tensors
                        # Store the list of input node names
                        linear_bn_inputs[node.target] = [str(inp) for inp in cat_inputs]
                else:
                    # Handle unexpected cases with multiple inputs
                    linear_bn_inputs[node.target] = [str(inp) for inp in input_nodes]

    return linear_bn_inputs
