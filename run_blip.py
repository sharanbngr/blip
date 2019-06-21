from __future__ import division
import json, pdb
import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import sys, ConfigParser, subprocess
from src.makeLISAdata import LISAdata
from src.bayes import Bayes
from tools.plotmaker import plotmaker
import matplotlib.pyplot as plt

class LISA(LISAdata, Bayes):

    '''
    Generic class for getting data and setting up the prior space and likelihood. This is tuned for ISGWB analysis at the moment
    but it should not be difficult to modify for other use cases. 
    '''

    def __init__(self,  params, inj):

        # set up the LISAdata class
        LISAdata.__init__(self, params, inj)
        # Set up the Bayes class
        Bayes.__init__(self)

        ## Make noise spectra
        self.which_noise_spectrum()
        self.which_response()

        ## Generate or get mldc data
        if self.params['mldc']:
            self.read_mldc_data()
        else:
            self.makedata()
        
        self.diag_spectra()


    def makedata(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to generate data. Return
        Frequency domain data. 
        '''

        ## Generate TDI noise
        h1, h2, h3 = self.gen_noise_spectrum()
    
        ## Generate TDI isotropic signal
        if self.inj['doInj']:
            h1_gw, h2_gw, h3_gw = self.gen_aet_isgwb()
            h1, h2, h3 = h1 + h1_gw, h2 + h2_gw, h3 + h3_gw

        ## Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata = self.tser2fser(h1, h2, h3)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)

    def read_mldc_data(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to read data. Return frequency
        domain data. Since this was used primarily for the MLDC, this assumes that the data is doppler tracking
        and converts to strain data. 
        '''
        
        h1, h2, h3 = self.read_data()
        
        ## Calculate other tdi combinations if necessary. 
        if self.params['tdi_lev']=='aet':
            h1 = (1.0/3.0)*(2*h1 - h2 - h3)
            h2 = (1.0/np.sqrt(3.0))*(h3 - h2)
            h3 = (1.0/3.0)*(h1 + h2 + h3)



        ## Generate lisa freq domain data from time domain data
        r1, r2, r3, self.fdata = self.tser2fser(h1, h2, h3)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)
        
        self.r1, self.r2, self.r3 = r1/(4*self.f0.reshape(self.f0.size, 1)), r2/(4*self.f0.reshape(self.f0.size, 1)), r3/(4*self.f0.reshape(self.f0.size, 1))
        

    def which_noise_spectrum(self):

        ## Figure out which instrumental noise spectra to use

        if self.params['tdi_lev']=='aet':
            self.instr_noise_spectrum = self.aet_noise_spectrum 
            self.gen_noise_spectrum = self.gen_aet_noise
        elif self.params['tdi_lev']=='xyz':
            self.instr_noise_spectrum = self.xyz_noise_spectrum
            self.gen_noise_spectrum = self.gen_xyz_noise
        elif self.params['tdi_lev']=='michelson':
            self.instr_noise_spectrum = self.mich_noise_spectrum
            self.gen_noise_spectrum = self.gen_michelson_noise

    def which_response(self):
    
        ## Figure out which antenna patterns to use

        if self.params['modeltype'] == 'isgwb' and params['tdi_lev']=='aet':
            self.R1, self.R2, self.R3 = self.isgwb_aet_response(self.f0)
        elif self.params['modeltype'] == 'isgwb' and params['tdi_lev']=='xyz':
            self.R1, self.R2, self.R3 = self.isgwb_xyz_response(self.f0)
        elif self.params['modeltype'] == 'isgwb' and params['tdi_lev']=='michelson':
            self.R1, self.R2, self.R3 = self.isgwb_mich_response(self.f0)  
        elif self.params['modeltype']=='sph_sgwb' and self.params['tdi_lev']=='aet':
            self.R1, self.R2, self.R3 = self.asgwb_aet_response(self.f0)
        elif self.params['modeltype'] == 'noise_only':
            print('Noise only model chosen ...')
        else:       
           raise ValueError('Unknown recovery model selected')


    def diag_spectra(self):

        '''
        A function to do simple diagnostics. Plot the expected spectra and data. 
        '''

        import scipy.signal as sg

        ## ------------ Calculate PSD ------------------
 
        # Number of segmants
    
        Nperseg=int(self.params['fs']*self.params['seglen'])

        ## PSD from the FFTs
        data_PSD = np.mean(np.abs(self.r1)**2, axis=1)

        # "Cut" to desired frequencies
        idx = np.logical_and(self.fdata >=  self.params['fmin'] , self.fdata <=  self.params['fmax'])
        psdfreqs = self.fdata[idx]

        #Charactersitic frequency
        fstar = 3e8/(2*np.pi*self.armlength)

        # define f0 = f/2f*
        f0 = self.fdata/(2*fstar)

        # Get desired frequencies for the PSD
        # We want to normalize PSDs to account for the windowing
        # Also convert from doppler-shift spectra to strain spectra
        data_PSD = data_PSD[idx]

        truevals = self.params['truevals']
        ## The last two elements are the position and the acceleration noise levels. 
        Np, Na = 10**truevals[-2], 10**truevals[-1]

        # Modelled Noise PSD
        S1, S2, S3 = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)        

        if self.params['modeltype'] != 'noise_only':
            ## SGWB signal levels of the mldc data
            Omega0, alpha = 10**truevals[1], truevals[0]

            ## Hubble constant
            H0 = 2.2*10**(-18)

            ## Calculate astrophysical power law noise
            Omegaf = Omega0*(self.fdata/25)**alpha

            ## Power spectra of the SGWB
            Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)
        
            ## Spectrum of the SGWB signal convoluted with the detector response tensor.
            S1_gw = Sgw*self.R1 

            ## The total noise spectra is the sum of the instrumental + astrophysical 
            S1 = S1+ S1_gw

            plt.loglog(self.fdata, 0.5*S1_gw, label='gw required')

        
        ## Plot data PSD with the expected level SAA
        plt.loglog(self.fdata, S1, label='required')

        plt.loglog(psdfreqs, data_PSD,label='PSD of the data series', alpha=0.6)
        fmin, fmax = 1e-4, 1e-1
        ymin, ymax = 1e-45, 1e-38
        plt.xlim(fmin, fmax)
        plt.ylim(ymin, ymax)
        plt.xlabel('f in Hz')
        plt.ylabel('Power Spectrum ')
        plt.legend()
        plt.savefig(self.params['out_dir'] + '/diag_psd.png', dpi=125)
        print('Diagnostic spectra plot made in ' + self.params['out_dir'] + '/diag_psd.png')
        import pdb; pdb.set_trace()
        plt.close()
        


def blip(paramsfile='params.ini'):
    '''
    The main workhorse of the bayesian pipeline.

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
    params['mldc'] = int(config.get("params", "mldc"))
    params['datafile']  = str(config.get("params", "datafile"))
    params['fref'] = float(config.get("params", "fref"))
    params['modeltype'] = str(config.get("params", "modeltype"))
    params['lmax'] = int(config.get("params", "lmax"))
    params['tdi_lev'] = str(config.get("params", "tdi_lev"))


    ## Extract truevals if any
    tlist = config.get('params', 'truevals')
    if len(tlist) >0:
        tvals = tlist.split(',')
        params['truevals'] = [float(tval) for tval in tvals] 
    else:
        params['truevals'] = []

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

    

    if params['modeltype']=='isgwb':

        print "Doing an isotropic stochastic analysis ..."
        parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$', r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
        npar = len(parameters)     
        engine = NestedSampler(lisa.isgwb_log_likelihood, lisa.isgwb_prior,\
                 npar, bound='multi', sample='rwalk', nlive=nlive)

    elif params['modeltype']=='sph_sgwb':

        print "Doing a spherical harmonic stochastic analysis ..."
        parameters = []

        parameters.append(r'$\alpha$')

        for ii in range(params['lmax'] + 1):
            omega_params = r'$\log_{10} (\Omega_' + str(ii) + ')$'
            parameters.append(omega_params)
        
        parameters.append( r'$\log_{10} (Np)$')
        parameters.append( r'$\log_{10} (Na)$')

        npar = len(parameters)
        engine = NestedSampler(lisa.sph_log_likelihood, lisa.sph_prior,\
                 npar, bound='multi', sample='rwalk', nlive=nlive)

    elif params['modeltype']=='noise_only':
        print "Doing an instrumental noise only analysis ..."
        parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
        npar = len(parameters)     
        engine = NestedSampler(lisa.instr_log_likelihood,  lisa.instr_prior,\
                 npar, bound='multi', sample='rwalk', nlive=nlive)

    else:
        raise ValueError('Unknown recovery model selected')


    print "npar = " + str(npar)
    
    
    # Check to see if we have appropriate number of truevals
    if (len(params['truevals']) != npar) and (len(params['truevals']) != 0):
        raise ValueError('The length of the truevals given does not match \
                the number of parameters for the model' )



    # -------------------- Extract and Plot posteriors ---------------------------
    
    engine.run_nested(dlogz=0.5,print_progress=True )


    # re-scale weights to have a maximum of one
    res = engine.results
    weights = np.exp(res['logwt'] - res['logz'][-1])
    weights[-1] = 1 - np.sum(weights[0:-1])

    post_samples = resample_equal(res.samples, weights)

    # Save posteriors to file
    np.savetxt(params['out_dir'] + "/post_samples.txt",post_samples)
    
    print("\n Making posterior Plots ...")
    plotmaker(params, parameters, npar)
    
if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide (only) the params file as an argument')
    else:
        blip(sys.argv[1])
