.. blip documentation master file, created by
   sphinx-quickstart on Thu Mar 14 07:51:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BLIP!
=================


BLIP stands for Bayesian LISA Pipeline and is a gravitational wave analysis pipeline for LISA. It is geared
mostly towards stochastic gravitational wave backgrounds, although the methods in here can be of wider use. 



Installation
------------

The code is fully pythonic and it is easiest to run this in a virtual environment. Make a new virtualenv lisawork by doing

``virtualenv lisawork``

Source it by

``source lisawork/bin/activate``

numpy, scipy are needed for running this. Matplotlib and corner are required for plotting. Install them all by doing

``pip install numpy scipy matplotlib corner``

We will also use  `dynesty <https://dynesty.readthedocs.io/en/latest/>`_  for the analysis. Dynesty is a nested sampler which gives out both the posteriors and bayesian evidence from it. Install dynesty by doing

``pip install dynesty``


Running the pipeline
---------------------
The code comes with a params.ini file which has all the parameters set in. You can change them and the signal model by modifying params.ini

To run the pipeline, do 

``python run_blip.py params.ini``



Support
-------

If you are having issues, please email banag002@umn.edu





Modules and Methods
------------------------

.. toctree::
   :maxdepth: 3
   
 
   classes
   aux_functions
    




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
