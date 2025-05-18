import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_grad_enabled(False)

# Define BatchNorm2d layer
bn_layer = nn.BatchNorm2d(num_features=10)
bn_layer.eval()

# Input tensor
dummy = torch.randn(1, 10, 8, 8, dtype=torch.float32)  # Match bn_layer's float32
y1 = bn_layer(dummy)  # Ground truth, float32

# Extract BatchNorm parameters
running_mean = bn_layer.running_mean  # Shape [10], float32
running_var = bn_layer.running_var   # Shape [10], float32
weight = bn_layer.weight             # Shape [10], float32
bias = bn_layer.bias                 # Shape [10], float32
eps = bn_layer.eps                   # 1e-05

# Debug: Print parameters
print(f"BN running_mean: {running_mean.tolist()}")
print(f"BN running_var: {running_var.tolist()}")
print(f"BN weight (gamma): {weight.tolist()}")
print(f"BN bias (beta): {bias.tolist()}")
print(f"BN eps: {eps}")

# Manual BatchNorm computation (vectorized, float32)
scale = weight / torch.sqrt(running_var + eps)  # Shape [10], float32
manual_out = (dummy - running_mean.view(1, -1, 1, 1)) * scale.view(1, -1, 1, 1) + bias.view(1, -1, 1, 1)

# Compute using F.batch_norm for validation
pytorch_out = F.batch_norm(
    dummy,
    running_mean,
    running_var,
    weight=weight,
    bias=bias,
    training=False,
    eps=eps
)

# Compute differences
diff_manual = torch.linalg.norm(y1 - manual_out)
relative_diff_manual = diff_manual / (torch.linalg.norm(y1) + 1e-10)
diff_pytorch = torch.linalg.norm(y1 - pytorch_out)
relative_diff_pytorch = diff_pytorch / (torch.linalg.norm(y1) + 1e-10)

print(f"Manual Difference (L2 norm): {diff_manual.item():.20f}")
print(f"Manual Relative difference: {relative_diff_manual.item():.20f}")
print(f"PyTorch F.batch_norm Difference (L2 norm): {diff_pytorch.item():.20f}")
print(f"PyTorch F.batch_norm Relative difference: {relative_diff_pytorch.item():.20f}")

# Per-channel differences
for c in range(10):
    channel_diff = torch.linalg.norm(y1[:, c, :, :] - manual_out[:, c, :, :])
    if channel_diff > 1e-10:
        print(f"Channel {c} difference: {channel_diff.item():.20f}")

# Debug: Check scale factor
expected_scale = 1.0 / torch.sqrt(torch.tensor(1.0 + 1e-5, dtype=torch.float32))
print(f"Expected scale: {expected_scale.item():.20f}")
print(f"Computed scale (first channel): {scale[0].item():.20f}")