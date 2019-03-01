from __future__ import division
import json, pdb
import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import sys, ConfigParser, subprocess
from src.makeLISAdata import LISAdata
from src.plotmaker import plotmaker

class LISA(LISAdata):

    def __init__(self,  params, inj):

        # set the data
        LISAdata.__init__(self, params, inj)
        
    '''
    def prior(self, theta):

        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0, log_Np, log_Na = theta

        # Transform to actual priors
        alpha       = 10*alpha-5
        log_omega0   = 6*log_omega0 -14
        log_Np = 5*log_Np - 44
        log_Na = 5*log_Na - 51

        return (alpha, log_omega0, log_Np, log_Na)



    def log_likelihood(self, theta):

        # Wrapper for isotropic loglikelihood
        llike = logL_iso(self.rA, self.rE, self.rT, self.fdata,  self.params, theta)
        return llike

    '''



def stochastic(paramsfile='params.ini'):
    '''
    The main workhorse of the stochastic pipeline.

    Input:
    Params File

    Output: Files containing evidence and pdfs of the parameters
    '''

    #  --------------- Read the params file --------------------------------

    # Initialize Dictionaries
    params = {}
    inj = {}

    config = ConfigParser.ConfigParser()
    config.read(paramsfile)

    # Params Dict
    params['fmin']     = float(config.get("params", "fmin"))
    params['fmax']     = float(config.get("params", "fmax"))
    params['dur']      = float(config.get("params", "duration"))
    params['seglen']   = float(config.get("params", "seglen"))
    params['fs']       = float(config.get("params", "fs"))
    params['Shfile']   = config.get("params", "Shfile")
    params['out_dir']            = str(config.get("run_params", "out_dir"))
    params['doPreProc']          = int(config.get("run_params", "doPreProc"))
    params['input_spectrum']     = str(config.get("run_params", "input_spectrum"))

    # Injection Dict
    inj['doInj']       = int(config.get("inj", "doInj"))
    inj['injtype']     = str(config.get("inj", "injtype"))
    inj['ln_omega0']   = np.log10(float(config.get("inj", "omega0")))
    inj['alpha']       = float(config.get("inj", "alpha"))
    inj['log_Np']      = np.log10(float(config.get("inj", "Np")))
    inj['log_Na']      = np.log10(float(config.get("inj", "Na")))

    # some run parameters
    
    verbose            = int(config.get("run_params", "verbose"))
    nlive              = int(config.get("run_params", "nlive"))
    nthread            = int(config.get("run_params", "Nthreads"))


    # --------------------------- NESTED SAMPLER --------------------------------

    # Make output folder
    subprocess.call(["mkdir", "-p", params['out_dir']])

    # Copy the params file to outdir, to keep track of the parameters of each run.
    subprocess.call(["cp", paramsfile, params['out_dir']])


    # ------------------------------ Run Nestle ----------------------------------

    # Initialize lisa class
    lisa =  LISA(params, inj)

    ## Generate TDI noise
    lisa.gen_aet_noise()

    ## Generate TDI isotropic signal
    lisa.gen_aet_isgwb()
    
    ## Generate lisa freq domain data from time domain data
    lisa.tser2fser()

    import pdb; pdb.set_trace()
    # Names of parameters

    '''
    parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$', r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
    npar = len(parameters)
    print "npar = " + str(npar)

    engine = NestedSampler(lisa.log_likelihood, lisa.prior, npar, bound='multi', sample='rwalk', nlive=nlive)
    engine.run_nested(dlogz=0.5,print_progress=True )
    #logL(rA, rE, rT, fdata,params, inj)



    # -------------------- Extract and Plot posteriors ---------------------------


    # re-scale weights to have a maximum of one
    res = engine.results
    weights = np.exp(res['logwt'] - res['logz'][-1])
    post_samples = resample_equal(res.samples, weights)

    # Save posteriors to file
    np.savetxt(out_dir + "/post_samples.txt",post_samples)


    print "\n Making posterior Plots ..."
    plotmaker(out_dir,post_samples, parameters, npar, inj)
    '''
if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide (only) the params file as an argument')
    else:
        stochastic(sys.argv[1])
