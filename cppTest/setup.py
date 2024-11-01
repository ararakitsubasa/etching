from setuptools import setup, Extension
import numpy as np

module = Extension('concatenate_module',
                   sources=['concatenate.cpp'],
                   include_dirs=[np.get_include()],
                   language='c++')

setup(name='concatenate_module',
      version='1.0',
      ext_modules=[module])
