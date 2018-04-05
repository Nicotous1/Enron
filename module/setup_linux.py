from distutils.core import setup
from Cython.Build import cythonize

import numpy
print(numpy.get_include())

setup(
    name = "SVBM",
    ext_modules = cythonize('SVBM.pyx', include_path = [numpy.get_include()]),  # accepts a glob pattern
)
