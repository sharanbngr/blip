#  BLIP: Bayesian LISA Inference Package

This is a fully Bayesian Python package for detecting/characterizing stochastic gravitational wave backgrounds and foregrounds with LISA.


1) We recommend creating a dedicated conda environment for BLIP. Conda is a common python virtual environment manager; if you already have Conda, start at step 2; otherwise [install conda.](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

2) Create an environment. We require Python 3.10 or later:

`conda create --name blip-env python=3.10`

3) Use the provided requirements yaml to install the remainder.

`conda env update -n blip-env --file=requirements.yml`

4) Activate it via

`conda activate blip-env`

5) You can now install the package via pip by running

`pip install .`

in this directory.

## JAX/NUMPYRO DEV BRANCH NOTE
As of 5/08/2023, only the git-installed dev version of Numpyro has checkpointing support. See Numpyro docs for git installation instructions. Requires newest version of jax (0.4.3). Numpyro installed through standard channels should work in most cases, but checkpointing will fail.


You should now be ready to go! To run BLIP, you only need to provide a configuration file. In this directory, you will find params_default.ini, a pre-constructed config file with reasonable settings and accompanying parameter explanations.

To run, call

`run_blip params_default.ini`

This will (by default) inject and recover a power law isotropic SGWB, with LISA detector noise at the level specified in the LISA proposal (Amaro-Seoane et al., 2017), for 3 months of data.

Two other helpful parameter files are also included: test_params.ini, which has settings ideal for (more) rapid code testing, and minimal_params.ini, which only includes the bare bones, minimal necessary settings for BLIP to run.

Posterior plots will be automatically created in the specified output directory, along with some diagnostics. All statistical model information is saved in Model.pickle; all information used to perform the injection is likewise saved in Injection.pickle. The posterior samples are saved to post_samples.txt.

More details can be found in [the code documentation](https://blip.readthedocs.io/en/latest/).