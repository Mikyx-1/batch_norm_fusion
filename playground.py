# ------------------------------------------------------------
# Author: Grok, based on Viet Hoang Le (Mikyx-1)'s suggestion
# Date: May 16, 2025
# Description:
# This script demonstrates fusing a Conv2d and BatchNorm2d layer and optimizing
# the fused weights using torch.optim over a batch of inputs to ensure objectivity.
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
    with torch.no_grad():
        # Try PyTorch's built-in fusion
        if conv.groups == 1:
            try:
                from torch.nn.utils import fuse_conv_bn_eval

                return fuse_conv_bn_eval(conv, bn)
            except ImportError:
                pass

        fusedconv = torch.nn.Conv2d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        ).to(dtype=torch.float64)

        w_conv = conv.weight.clone().to(dtype=torch.float64)
        bn_eps = 1e-12

        var_stable = (
            torch.log1p(bn.running_var.to(dtype=torch.float64).clamp(min=0)) + bn_eps
        )
        scale_factor = bn.weight.to(dtype=torch.float64) / torch.sqrt(var_stable)

        if conv.groups > 1:
            fused_weight = torch.zeros_like(w_conv, dtype=torch.float64)
            group_size = conv.out_channels // conv.groups
            for g in range(conv.groups):
                w_conv_g = w_conv[g * group_size : (g + 1) * group_size]
                w_bn_g = torch.diag(scale_factor[g * group_size : (g + 1) * group_size])
                fused_weight[g * group_size : (g + 1) * group_size] = torch.matmul(
                    w_bn_g, w_conv_g.view(group_size, -1)
                ).view_as(w_conv_g)
        else:
            w_bn = torch.diag(scale_factor)
            fused_weight = torch.matmul(w_bn, w_conv.view(conv.out_channels, -1)).view(
                fusedconv.weight.size()
            )

        fused_weight = torch.tanh(fused_weight / 1e4) * 1e4
        fusedconv.weight.copy_(fused_weight)

        if conv.bias is not None:
            b_conv = conv.bias.clone().to(dtype=torch.float64)
        else:
            b_conv = torch.zeros(
                conv.out_channels, device=conv.weight.device, dtype=torch.float64
            )

        b_bn = bn.bias.to(dtype=torch.float64) - bn.weight.to(dtype=torch.float64).mul(
            bn.running_mean.to(dtype=torch.float64)
        ).div(torch.sqrt(var_stable))
        fused_bias = torch.matmul(w_bn, b_conv.unsqueeze(1)).squeeze() + b_bn
        fused_bias = torch.tanh(fused_bias / 1e4) * 1e4
        fusedconv.bias.copy_(fused_bias)

    return fusedconv


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


def main():
    torch.manual_seed(42)

    model = ConvBNModel().eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.to(dtype=torch.float64)  # Convert model parameters to float64

    # Stabilize BN with normalized inputs
    for _ in range(100):
        input_tensor = torch.randn(64, 3, 16, 16, device=device, dtype=torch.float64)
        input_tensor = input_tensor / (
            torch.amax(input_tensor.abs(), dim=(1, 2, 3), keepdim=True) + 1e-10
        )
        model(input_tensor)

    # Create training batch
    batch_size = 64
    train_inputs_small = torch.randn(
        batch_size, 3, 16, 16, device=device, dtype=torch.float64
    )
    train_inputs_small = train_inputs_small / (
        torch.amax(train_inputs_small.abs(), dim=(1, 2, 3), keepdim=True) + 1e-10
    )
    train_inputs_large = torch.randn(
        batch_size, 3, 224, 224, device=device, dtype=torch.float64
    )
    train_inputs_large = train_inputs_large / (
        torch.amax(train_inputs_large.abs(), dim=(1, 2, 3), keepdim=True) + 1e-10
    )

    # Create validation batch
    val_inputs_small = torch.randn(
        batch_size, 3, 16, 16, device=device, dtype=torch.float64
    )
    val_inputs_small = val_inputs_small / (
        torch.amax(val_inputs_small.abs(), dim=(1, 2, 3), keepdim=True) + 1e-10
    )
    val_inputs_large = torch.randn(
        batch_size, 3, 224, 224, device=device, dtype=torch.float64
    )
    val_inputs_large = val_inputs_large / (
        torch.amax(val_inputs_large.abs(), dim=(1, 2, 3), keepdim=True) + 1e-10
    )

    # Get original outputs
    with torch.no_grad():
        original_outputs_small = model(train_inputs_small)
        original_outputs_large = model(train_inputs_large)
        val_original_small = model(val_inputs_small)
        val_original_large = model(val_inputs_large)

    fused_layer = fuse_conv_and_bn(model.conv, model.bn)
    fused_layer.eval()
    fused_layer.to(device)

    # Get output before optimization
    with torch.no_grad():
        fused_outputs_small = fused_layer(train_inputs_small)
        fused_outputs_large = fused_layer(train_inputs_large)
        val_fused_small = fused_layer(val_inputs_small)
        val_fused_large = fused_layer(val_inputs_large)
    diff_before_small = (
        torch.linalg.norm(original_outputs_small - fused_outputs_small) / batch_size
    )
    diff_before_large = (
        torch.linalg.norm(original_outputs_large - fused_outputs_large) / batch_size
    )
    val_diff_before_small = (
        torch.linalg.norm(val_original_small - val_fused_small) / batch_size
    )
    val_diff_before_large = (
        torch.linalg.norm(val_original_large - val_fused_large) / batch_size
    )
    print(f"Train difference before optimization (16x16): {diff_before_small:.6f}")
    print(f"Train difference before optimization (224x224): {diff_before_large:.6f}")
    print(f"Val difference before optimization (16x16): {val_diff_before_small:.6f}")
    print(f"Val difference before optimization (224x224): {val_diff_before_large:.6f}")

    # Optimize fused layer
    fused_layer = fused_layer.to(dtype=torch.float64)
    fused_layer.weight.requires_grad_(True)
    fused_layer.bias.requires_grad_(True)
    initial_weight = fused_layer.weight.clone().detach()
    initial_bias = fused_layer.bias.clone().detach()
    optimizer = optim.LBFGS(
        [fused_layer.weight, fused_layer.bias],
        lr=1e-4,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        fused_outputs = fused_layer(train_inputs_small)
        # Compute L2 norm per sample across channels, height, width
        fused_norm = (
            torch.sqrt(torch.sum(fused_outputs**2, dim=(1, 2, 3), keepdim=True)) + 1e-10
        )
        fused_outputs_norm = fused_outputs / fused_norm
        original_norm = (
            torch.sqrt(
                torch.sum(original_outputs_small**2, dim=(1, 2, 3), keepdim=True)
            )
            + 1e-10
        )
        original_outputs_norm = original_outputs_small / original_norm
        loss_output = (
            torch.linalg.norm(fused_outputs_norm - original_outputs_norm) / batch_size
        )
        loss_weight = 1e-3 * torch.linalg.norm(fused_layer.weight - initial_weight)
        loss_bias = 1e-3 * torch.linalg.norm(fused_layer.bias - initial_bias)
        loss = loss_output + loss_weight + loss_bias
        loss.backward()
        return loss

    num_iterations = 1000
    for i in range(num_iterations):
        optimizer.step(closure)
        with torch.no_grad():
            fused_outputs = fused_layer(train_inputs_small)
            fused_norm = (
                torch.sqrt(torch.sum(fused_outputs**2, dim=(1, 2, 3), keepdim=True))
                + 1e-10
            )
            fused_outputs_norm = fused_outputs / fused_norm
            original_norm = (
                torch.sqrt(
                    torch.sum(original_outputs_small**2, dim=(1, 2, 3), keepdim=True)
                )
                + 1e-10
            )
            original_outputs_norm = original_outputs_small / original_norm
            loss_output = (
                torch.linalg.norm(fused_outputs_norm - original_outputs_norm)
                / batch_size
            )
        if i % 100 == 0:
            print(f"Iteration {i}, Output Loss: {loss_output.item():.6f}")

    # Convert to float32
    with torch.no_grad():
        fused_layer = fused_layer.to(dtype=torch.float32)
        fused_layer.weight.copy_(fused_layer.weight.to(dtype=torch.float32))
        fused_layer.bias.copy_(fused_layer.bias.to(dtype=torch.float32))
        print(
            f"Weight dtype: {fused_layer.weight.dtype}, Bias dtype: {fused_layer.bias.dtype}"
        )

    fused_layer.weight.requires_grad_(False)
    fused_layer.bias.requires_grad_(False)

    # Get output after optimization
    with torch.no_grad():
        train_inputs_small = train_inputs_small.to(dtype=torch.float32)
        train_inputs_large = train_inputs_large.to(dtype=torch.float32)
        val_inputs_small = val_inputs_small.to(dtype=torch.float32)
        val_inputs_large = val_inputs_large.to(dtype=torch.float32)
        fused_outputs_small = fused_layer(train_inputs_small)
        fused_outputs_large = fused_layer(train_inputs_large)
        val_fused_small = fused_layer(val_inputs_small)
        val_fused_large = fused_layer(val_inputs_large)
    diff_after_small = (
        torch.linalg.norm(
            original_outputs_small.to(dtype=torch.float32) - fused_outputs_small
        )
        / batch_size
    )
    diff_after_large = (
        torch.linalg.norm(
            original_outputs_large.to(dtype=torch.float32) - fused_outputs_large
        )
        / batch_size
    )
    val_diff_after_small = (
        torch.linalg.norm(val_original_small.to(dtype=torch.float32) - val_fused_small)
        / batch_size
    )
    val_diff_after_large = (
        torch.linalg.norm(val_original_large.to(dtype=torch.float32) - val_fused_large)
        / batch_size
    )
    print(f"Train difference after optimization (16x16): {diff_after_small:.6f}")
    print(f"Train difference after optimization (224x224): {diff_after_large:.6f}")
    print(f"Val difference after optimization (16x16): {val_diff_after_small:.6f}")
    print(f"Val difference after optimization (224x224): {val_diff_after_large:.6f}")

    if diff_after_small < 1e-6:
        print("Success: Train difference (16x16) is less than 1e-6!")
    else:
        print("Warning: Train difference (16x16) is still above 1e-6.")
    if diff_after_large < 1e-6:
        print("Success: Train difference (224x224) is less than 1e-6!")
    else:
        print("Warning: Train difference (224x224) is still above 1e-6.")
    if val_diff_after_small < 1e-6:
        print("Success: Val difference (16x16) is less than 1e-6!")
    else:
        print("Warning: Val difference (16x16) is still above 1e-6.")
    if val_diff_after_large < 1e-6:
        print("Success: Val difference (224x224) is less than 1e-6!")
    else:
        print("Warning: Val difference (224x224) is still above 1e-6.")


if __name__ == "__main__":
    main()
