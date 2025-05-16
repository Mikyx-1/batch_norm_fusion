# ------------------------------------------------------------
# Author: Grok, based on Viet Hoang Le (Mikyx-1)'s suggestion
# Date: May 16, 2025
# Description:
# This script demonstrates fusing a Conv2d and BatchNorm2d layer and optimizing
# the fused weights using torch.optim to match the original output.
# License: MIT License
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim


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
    # Try PyTorch's built-in fusion for standard convolutions
    if conv.groups == 1:
        try:
            from torch.nn.utils import fuse_conv_bn_eval

            return fuse_conv_bn_eval(conv, bn)
        except ImportError:
            pass  # Fall back to custom fusion if utility is unavailable

    # Custom fusion for grouped convolutions or if PyTorch utility fails
    fusedconv = torch.nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).to(
        dtype=torch.float64
    )  # Use double precision

    w_conv = conv.weight.clone().to(dtype=torch.float64)
    if conv.groups > 1:
        # Process grouped convolutions group-wise
        fused_weight = torch.zeros_like(w_conv, dtype=torch.float64)
        group_size = conv.out_channels // conv.groups
        for g in range(conv.groups):
            w_conv_g = w_conv[g * group_size : (g + 1) * group_size]
            w_bn_g = torch.diag(
                bn.weight[g * group_size : (g + 1) * group_size]
                .to(dtype=torch.float64)
                .div(
                    torch.sqrt(
                        bn.running_var[g * group_size : (g + 1) * group_size].to(
                            dtype=torch.float64
                        )
                        + bn.eps
                        + 1e-4
                    )
                )
            )
            fused_weight[g * group_size : (g + 1) * group_size] = torch.mm(
                w_bn_g, w_conv_g.view(group_size, -1)
            ).view_as(w_conv_g)
    else:
        # Regular convolution
        w_bn = torch.diag(
            bn.weight.to(dtype=torch.float64).div(
                torch.sqrt(bn.running_var.to(dtype=torch.float64) + bn.eps + 1e-4)
            )
        )
        fused_weight = torch.mm(w_bn, w_conv.view(conv.out_channels, -1)).view(
            fusedconv.weight.size()
        )

    fusedconv.weight.copy_(
        fused_weight.to(dtype=torch.float32)
    )  # Convert back to float32

    if conv.bias is not None:
        b_conv = conv.bias.clone().to(dtype=torch.float64)
    else:
        b_conv = torch.zeros(
            conv.out_channels, device=conv.weight.device, dtype=torch.float64
        )

    b_bn = bn.bias.to(dtype=torch.float64) - bn.weight.to(dtype=torch.float64).mul(
        bn.running_mean.to(dtype=torch.float64)
    ).div(torch.sqrt(bn.running_var.to(dtype=torch.float64) + bn.eps + 1e-4))
    fusedconv.bias.copy_(
        (torch.matmul(w_bn, b_conv.unsqueeze(1)).squeeze() + b_bn).to(
            dtype=torch.float32
        )
    )

    return fusedconv.to(dtype=torch.float32)


# Define a simple Conv + BN model
class ConvBNModel(nn.Module):
    def __init__(self):
        super(ConvBNModel, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Main experiment
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Initialize model and stabilize BN statistics
    model = ConvBNModel().eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Stabilize BN with normalized inputs
    for _ in range(100):
        input_tensor = torch.randn(1, 3, 32, 32, device=device)
        input_tensor = input_tensor / (input_tensor.abs().max() + 1e-10)
        model(input_tensor)

    # Create test input
    test_input = torch.randn(1, 3, 224, 224, device=device)
    test_input = test_input / (test_input.abs().max() + 1e-10)

    # Get original output
    with torch.no_grad():
        original_output = model(test_input)

    # Fuse Conv and BN
    fused_layer = fuse_conv_and_bn(model.conv, model.bn)
    fused_layer.eval()
    fused_layer.to(device)

    # Get output before optimization
    with torch.no_grad():
        fused_output = fused_layer(test_input)
    diff_before = torch.linalg.norm(original_output - fused_output)
    print(f"Output difference before optimization: {diff_before:.6f}")

    # Optimize fused layer
    fused_layer.weight.requires_grad_(True)
    fused_layer.bias.requires_grad_(True)
    optimizer = optim.Adam([fused_layer.weight, fused_layer.bias], lr=1e-12)
    num_iterations = 1000

    for i in range(num_iterations):
        optimizer.zero_grad()
        fused_output = fused_layer(test_input)
        loss = torch.linalg.norm(fused_output - original_output)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.6f}")

    # Get output after optimization
    with torch.no_grad():
        fused_output = fused_layer(test_input)
    diff_after = torch.linalg.norm(original_output - fused_output)
    print(f"Output difference after optimization: {diff_after:.6f}")

    # Verify if difference is less than 1e-6
    if diff_after < 1e-6:
        print("Success: Output difference is less than 1e-6!")
    else:
        print("Warning: Output difference is still above 1e-6.")


if __name__ == "__main__":
    main()
