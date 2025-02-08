# Fuse BatchNorm into Conv for PyTorch Models

### **Author:** Viet Hoang Le (Mikyx-1)  
**Date:** December 8th, 2024  

---

## **Overview**
This repository provides a PyTorch implementation for fusing BatchNorm layers into Conv layers, enabling optimised model inference. By reducing redundant parameters and computations, this tool makes models faster and more memory-efficient without altering their functionality.

This script is designed for the deep learning community, addressing the need for streamlined deployment of neural networks, especially in resource-constrained environments.

---

## **Features**
- **Automatic BatchNorm Fusion**: Combines Conv2D and BatchNorm2D layers.
- **Support for Grouped Convolutions**: Handles both regular and grouped convolutions seamlessly.
- **Parameter Reduction Tracking**: Calculates and displays the total number of parameters reduced after fusion.
- **Simple Integration**: Easy-to-use functions that work on any PyTorch model.
- **Preserves Model Architecture**: Maintains compatibility with pre-trained weights and inference pipelines.

---

## **Why Use This Script?**
- **Speed**: Fused models run faster during inference due to reduced computation.
- **Efficiency**: Minimises memory usage by removing unnecessary parameters.
- **Seamless Integration**: Works with any PyTorch-based architecture.
- **Community-Focused**: Developed with simplicity and extensibility in mind.

---

## ⚠️ Warning & Precision Considerations  

BatchNorm fusion may introduce **small numerical differences** at every fusion step due to floating-point precision changes. These small errors can **accumulate**, leading to noticeable deviations in the final model output, especially in deeper networks.  

### **Recommended Steps to Validate Your Model:**  

#### **1. Compare Model Outputs Before and After Fusion**  

Use the following code to measure the relative error between the original and fused model outputs:  

```python
import torch

# Example input tensor
input_tensor = torch.randn(1, 3, 224, 224)  # Adjust shape based on your model's input

# Get outputs before and after fusion
orig_out = model(input_tensor)
fused_out = fused_model(input_tensor)

# Compute relative error
relative_error = torch.norm(orig_out - fused_out) / torch.norm(orig_out)
print(f"Relative Error: {relative_error.item():.6f}")
```

#### 2. **Test on a Validation Dataset**

Run inference on a validation set and compare the accuracy before and after fusion.

#### 3. **If Necessary, Adjust BatchNorm Parameters**

If precision is critical, consider adjusting eps values in BatchNorm layers or retaining BatchNorm layers selectively.

➡ **Always verify fusion results before deploying the fused model into production!**

## **Installation**
Clone the repository:

```bash
git clone https://github.com/Mikyx-1/batch_norm_fusion
cd batch_norm_fusion
```

---

## **How to Use**

### **1. Import the Library**
```python
from fusion import fuse
```

### **2. Fuse Your Model**

```python
# Import your model
from your_model import MyModel

# Initialise the model
model = MyModel()

# Fuse BatchNorm into Conv layers
fused_model = fuse(model)

# Save or use the fused model
torch.save(fused_model.state_dict(), "fused_model.pth")
```

### **3. Key Output**
- The function will display the total number of parameters reduced:
  ```
  BatchNorm fusion completed. 1250 parameters were reduced after fusion.
  ```

---

## **Function Descriptions**

### **`fuse_conv_and_bn`**
Fuses a single Conv2D layer with a BatchNorm2D layer. Handles grouped convolutions efficiently.

#### **Arguments:**
- `conv (torch.nn.Conv2d)`: The convolutional layer.
- `bn (torch.nn.BatchNorm2d)`: The BatchNorm layer.

#### **Returns:**
- `torch.nn.Conv2d`: The fused convolutional layer.

---

### **`fuse`**
Iterates through an entire model, fusing all eligible BatchNorm and Conv layers.

#### **Arguments:**
- `model (torch.nn.Module)`: The PyTorch model to be fused.

#### **Returns:**
- `torch.nn.Module`: A new model with BatchNorm layers fused into Conv layers.

---

## **Examples**

### **Before Fusion**
```python
MyModel(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

### **After Fusion**
```python
MyModel(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
  (bn1): Identity()
)
```

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
Special thanks to the deep learning community for inspiring this work. Contributions and feedback are always welcome!

---

## **Contribute**
If you encounter bugs or have suggestions for improvement, please open an issue or submit a pull request.

---

## **Contact**
For questions or collaboration, feel free to reach out:
- GitHub: [Mikyx-1](https://github.com/Mikyx-1)
- Email: lehoangviet2k@gmail.com

