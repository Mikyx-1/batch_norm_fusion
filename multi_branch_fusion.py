import torch
import torch.nn as nn
import torch.optim as optim
from fusion import fuse

torch.set_grad_enabled(False)

class Model(nn.Module):
    def __init__(self, in_channels: int = 16, out_channels: int = 32, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.bn = nn.BatchNorm2d(out_channels*3)

    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = torch.concat([x1, x2, x3], dim=1)
        x = self.bn(x)
        return x
    

if __name__ == "__main__":
    model = Model(in_channels=3)
    model.eval()

    fused_model = fuse(model)
    fused_model.eval()    
    print(fused_model)