## setup.py file to compile Cython modules

from distutils.core import setup
from Cython.Build import cythonize

setup(name='Orbiting ISGWB Detector Response', ext_modules=cythonize("isgwbresponse.pyx"))