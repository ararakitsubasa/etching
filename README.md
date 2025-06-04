# 项目名称

本项目依赖以下 Python 科学计算和可视化库：

- [NumPy](https://numpy.org/): 数值计算基础库
- [PyVista](https://docs.pyvista.org/): 3D 可视化与建模工具
- [Numba](https://numba.pydata.org/): JIT 编译加速 Python 数值计算
- [PyTorch](https://pytorch.org/): 机器学习框架
- [Matplotlib](https://matplotlib.org/): 数据可视化库

## 环境要求

- Python 3.8+（建议使用 [Anaconda](https://www.anaconda.com/) 环境）
- 建议在虚拟环境中安装，避免依赖冲突

## 安装方式

### 使用 conda（推荐）

```bash
conda create -n myenv python=3.10
conda activate myenv
conda install numpy matplotlib numba pyvista -c conda-forge
conda install pytorch torchvision torchaudio cpuonly -c pytorch  # CPU 版
```

run [ID242-3_CVD_depo_void_reflection.ipynb](https://github.com/ararakitsubasa/etching/blob/main/ID242-3_CVD_depo_void_reflection.ipynb) for test
