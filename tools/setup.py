from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

 
import numpy
import sys
 
include_gsl_dir = "/usr/local/include/"
lib_gsl_dir = "/usr/local/lib/"
     
ext = Extension("cython_response", ["./src/cython_func/cython_response.pyx"],
    include_dirs=[numpy.get_include(), 
                  include_gsl_dir],
    library_dirs=[lib_gsl_dir],
    libraries=["gsl"]
)
 
setup(ext_modules=[ext],
    cmdclass = {'build_ext': build_ext})

os.system('mv cython_response.so src/cython_func/') 

