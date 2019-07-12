from distutils.core import setup 
from Cython.Build import cythonize
    
setup(ext_modules=cythonize('./src/cython_func/cython_response.pyx'))
