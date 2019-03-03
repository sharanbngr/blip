from __future__ import division
import json, pdb
import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import sys, ConfigParser, subprocess
from src.makeLISAdata import LISAdata
from tools.plotmaker import plotmaker

class LISA(LISAdata):

    def __init__(self,  params, inj):

        # set the data
        LISAdata.__init__(self, params, inj)

        if self.params['readData']:
            self.readdata()
        else:
            self.makedata()


    def makedata(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to generate data. Return
        Frequency domain data. 
        '''

        ## Generate TDI noise
        h1, h2, h3 = self.gen_aet_noise()

        ## Generate TDI isotropic signal
        if self.inj['doInj']:
            h1_gw, h2_gw, h3_gw = self.gen_aet_isgwb()
            h1, h2, h3 = h1 + h1_gw, h2 + h2_gw, h3 + h3_gw

        ## Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata = self.tser2fser(h1, h2, h3)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = (self.fdata/(2*fstar)).reshape(self.fdata.size, 1)

    def readdata(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to read data. Return frequency
        domain data. Since this was used primarily for the MLDC, this assumes that the data is doppler tracking
        and converts to strain data. 
        '''
        
        h1, h2, h3 = self.read_data()
        
        ## Generate lisa freq domain data from time domain data
        r1, r2, r3, self.fdata = self.tser2fser(h1, h2, h3)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = (self.fdata/(2*fstar)).reshape(self.fdata.size, 1)

        self.r1, self.r2, self.r3 = r1/(4*self.f0), r2/(4*self.f0), r3/(4*self.f0)


        import pdb; pdb.set_trace()
        
   
   
    
    def isgwb_prior(self, theta):

        '''
        Prior function for the ISGWB
        '''

        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0, log_Np, log_Na = theta

        # Transform to actual priors
        alpha       = 10*alpha-5
        log_omega0   = 6*log_omega0 -14
        log_Np = 5*log_Np - 44
        log_Na = 5*log_Na - 51

        return (alpha, log_omega0, log_Np, log_Na)



    def isgwb_log_likelihood(self, theta):

        # Wrapper for isotropic loglikelihood
        llike = isgwb_logL(lisa, theta)
        return llike

    



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
    params['readData'] = int(config.get("params", "readData"))
    params['datafile']  = str(config.get("params", "datafile"))

    # Injection Dict
    inj['doInj']       = int(config.get("inj", "doInj"))
    inj['injtype']     = str(config.get("inj", "injtype"))
    inj['ln_omega0']   = np.log10(float(config.get("inj", "omega0")))
    inj['alpha']       = float(config.get("inj", "alpha"))
    inj['log_Np']      = np.log10(float(config.get("inj", "Np")))
    inj['log_Na']      = np.log10(float(config.get("inj", "Na")))

    # some run parameters
    params['out_dir']            = str(config.get("run_params", "out_dir"))
    params['doPreProc']          = int(config.get("run_params", "doPreProc"))
    params['input_spectrum']     = str(config.get("run_params", "input_spectrum"))
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
    
    # Names of parameters

    
    parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$', r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
    npar = len(parameters)
    print "npar = " + str(npar)

    engine = NestedSampler(lisa.isgwb_log_likelihood, lisa.isgwb_prior, npar, bound='multi', sample='rwalk', nlive=nlive)
    engine.run_nested(dlogz=0.5,print_progress=True )




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
