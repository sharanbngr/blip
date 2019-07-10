## setup.py file to compile Cython modules

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import shutil

setup(name='OrbitingISGWBResponse', ext_modules=cythonize("isgwbresponse.pyx", annotate=True),include_dirs=[np.get_include()])

shutil.move('isgwbresponse.so','src/isgwbresponse.so')