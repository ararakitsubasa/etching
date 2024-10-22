from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# 需要添加 OpenMP 的编译器标志
# extra_compile_args = ['-fopenmp']  # GCC 或 Clang
# extra_link_args = ['-fopenmp']      # GCC 或 Clang

# 对于 MSVC（Windows）
extra_compile_args = ['/openmp']   # 对于 MSVC
extra_link_args = ['/openmp']       # 对于 MSVC

setup(
    ext_modules=cythonize(
        "boundary.pyx",
        compiler_directives={'language_level': "3"},  # 指定语言级别
        annotate=True  # 生成注释文件，可以帮助调试
    ),
    include_dirs=[np.get_include()],  # 包含 NumPy 的头文件
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)
