from typing import Dict, List, Set

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from fusion import fuse

torch.set_grad_enabled(False)


def extract_bn_conv_pairs(gm: GraphModule) -> Dict[str, List[str]]:
    """
    For each BatchNorm node in the FX graph, walk backwards through its inputs
    and collect the names of all Conv modules (Conv1d/2d/3d) that eventually
    feed into it.

    Returns:
        { bn_node_name: [conv_module_name, ...],  ... }
    """
    modules = dict(gm.named_modules())
    pairs: Dict[str, List[str]] = {}

    def find_convs(node: Node, visited: Set[Node]) -> List[str]:
        if node in visited:
            return []
        visited.add(node)

        # If this node is directly a conv module call, capture it
        if node.op == "call_module" and isinstance(
            modules[node.target], (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        ):
            return [node.target]

        convs: List[str] = []
        # Otherwise, recurse into all Node args
        for arg in node.args:
            if isinstance(arg, Node):
                convs.extend(find_convs(arg, visited))
            elif isinstance(arg, (tuple, list)):
                # if the argument is a list/tuple of nodes, also recurse
                for elem in arg:
                    if isinstance(elem, Node):
                        convs.extend(find_convs(elem, visited))

        return convs

    for node in gm.graph.nodes:
        # find every BatchNorm node
        if node.op == "call_module" and isinstance(
            modules[node.target], (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ):
            bn_name = node.name
            # start traversal from its first input
            start = node.args[0]
            conv_names = find_convs(start, set())
            # dedupe while preserving order
            seen = set()
            conv_list = []
            for c in conv_names:
                if c not in seen:
                    seen.add(c)
                    conv_list.append(c)
            pairs[bn_name] = conv_list

    return pairs


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.bn = nn.BatchNorm2d(out_channels * 3)

    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = torch.concat([x1, x2, x3], dim=1)
        x = self.bn(x)
        return x


if __name__ == "__main__":
    import torchvision

    model = Model(in_channels=3)
    # model = torchvision.models.resnet18(weights=None)
    model.eval()
    gm = torch.fx.symbolic_trace(model)
    pairs = extract_bn_conv_pairs(gm)
    print(pairs)
