from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="conv_relu_cpp",
    ext_modules=[
        CppExtension(
            "conv_relu_cpp",
            ["conv_relu.cpp"],
            extra_compile_args=["-O3", "-march=native"],
            extra_link_args=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
