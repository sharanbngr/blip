#  BLIP: Bayesian LISA Pipeline

More documentation at https://sharanbngr.github.io/blip/


A bayesian pipeline for detecting stochastic backgrounds with LISA. BLIP stands for Bayesian LIsa Pipeline. This is a fully pythonic pipeline.

For now we have only the isotropic background. It is easiest to run this in a virtual environment. 
Make a new virtualenv lisawork by doing

`virtualenv lisawork`

Source it by

`source lisawork/bin/activate`

numpy, scipy are needed for running this. Matplotlib and corner are required for plotting. Install them all by doing

`pip install numpy scipy matplotlib corner`

The sampler [dynesty](https://dynesty.readthedocs.io/en/latest/) is used for nested sampling. We get both the posteriors and bayesian evidence from it. The latter is the detection statistic. Install dynesty by doing

`pip install dynesty`

You can change the parameters and the signal model in params.ini

To run do `python run_blip.py params.ini`

Posterior plots are automatically made in the output directory specified in params.ini

Note: Currently the code is setup to run with python 2.7 and might not work with python 3