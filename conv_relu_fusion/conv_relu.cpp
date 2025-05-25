#include <torch/torch.h>
#include <c10/util/Exception.h>
#include <omp.h>

// Fused Conv2d + ReLU forward function
at::Tensor conv_relu_forward(
    const at::Tensor& input,      // [batch, in_channels, height, width]
    const at::Tensor& weight,     // [out_channels, in_channels, kernel_size, kernel_size]
    const at::Tensor& bias,       // [out_channels]
    int stride,                   // Stride
    int padding)                  // Padding
{
    // Validate inputs
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (batch, in_channels, height, width)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (out_channels, in_channels, kernel_size, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D (out_channels)");
    TORCH_CHECK(input.is_cpu() && weight.is_cpu() && bias.is_cpu(), "All tensors must be on CPU");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32,
                "Input, weight, and bias must be float32");

    // Ensure input and weight are contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Get dimensions
    int batch_size = input_contig.size(0);
    int in_channels = input_contig.size(1);
    int height = input_contig.size(2);
    int width = input_contig.size(3);
    int out_channels = weight_contig.size(0);
    int kernel_size = weight_contig.size(2); // Assume square kernel
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;

    // Allocate output tensor
    auto output = torch::empty({batch_size, out_channels, out_height, out_width}, input.options());

    // Access raw data
    auto input_data = input_contig.data_ptr<float>();
    auto weight_data = weight_contig.data_ptr<float>();
    auto bias_data = bias_contig.data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    // Perform fused Conv + ReLU with parallelization
    #pragma omp parallel
    {
        // Parallelize batch loop for small batch sizes; fallback to output channels if batch_size=1
        if (batch_size > 1) {
            #pragma omp for schedule(dynamic)
            for (int b = 0; b < batch_size; ++b) {
                for (int oc = 0; oc < out_channels; ++oc) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
                            float sum = bias_data[oc];
                            for (int ic = 0; ic < in_channels; ++ic) {
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                            sum += input_data[b * in_channels * height * width +
                                                             ic * height * width + ih * width + iw] *
                                                   weight_data[oc * in_channels * kernel_size * kernel_size +
                                                               ic * kernel_size * kernel_size + kh * kernel_size + kw];
                                        }
                                    }
                                }
                            }
                            // Apply ReLU directly
                            output_data[b * out_channels * out_height * out_width +
                                       oc * out_height * out_width + oh * out_width + ow] = std::max(0.0f, sum);
                        }
                    }
                }
            }
        } else {
            #pragma omp for schedule(dynamic)
            for (int oc = 0; oc < out_channels; ++oc) {
                for (int b = 0; b < batch_size; ++b) {
                    for (int oh = 0; oh < out_height; ++oh) {
                        for (int ow = 0; ow < out_width; ++ow) {
                            float sum = bias_data[oc];
                            for (int ic = 0; ic < in_channels; ++ic) {
                                for (int kh = 0; kh < kernel_size; ++kh) {
                                    for (int kw = 0; kw < kernel_size; ++kw) {
                                        int ih = oh * stride - padding + kh;
                                        int iw = ow * stride - padding + kw;
                                        if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                                            sum += input_data[b * in_channels * height * width +
                                                             ic * height * width + ih * width + iw] *
                                                   weight_data[oc * in_channels * kernel_size * kernel_size +
                                                               ic * kernel_size * kernel_size + kh * kernel_size + kw];
                                        }
                                    }
                                }
                            }
                            // Apply ReLU directly
                            output_data[b * out_channels * out_height * out_width +
                                       oc * out_height * out_width + oh * out_width + ow] = std::max(0.0f, sum);
                        }
                    }
                }
            }
        }
    }

    return output;
}

// Register the function for Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_relu_forward", &conv_relu_forward, "Fused Conv2d + ReLU forward");
}