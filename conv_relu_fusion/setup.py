from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# Enable OpenMP support based on platform
extra_compile_args = ["-O3", "-march=native", "-fopenmp"]
extra_link_args = ["-fopenmp"]

# Optional: Add include and library paths for libtorch if needed
# Update these paths based on your system
extra_include_paths = []
extra_library_paths = []

# Check platform (Linux/macOS/Windows)
import platform
if platform.system() == "Windows":
    # OpenMP on Windows with MSVC requires different flags
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = ["/openmp"]
elif platform.system() == "Darwin":  # macOS
    # macOS often uses clang, which may require libomp
    extra_compile_args = ["-O3", "-march=native", "-Xpreprocessor", "-fopenmp"]
    extra_link_args = ["-lomp"]
    # Ensure Homebrew's libomp is available if installed
    extra_include_paths = ["/usr/local/opt/libomp/include"]
    extra_library_paths = ["/usr/local/opt/libomp/lib"]

setup(
    name="conv_relu_cpp",
    ext_modules=[
        CppExtension(
            "conv_relu_cpp",
            ["conv_relu.cpp"],  # Ensure this matches your source file name
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=extra_include_paths,
            library_dirs=extra_library_paths,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)