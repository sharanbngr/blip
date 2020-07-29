import pickle
import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import sys, configparser, subprocess
from src.makeLISAdata import LISAdata
from src.bayes import Bayes
from tools.plotmaker import plotmaker
import matplotlib.pyplot as plt
# from eogtest import open_img


class LISA(LISAdata, Bayes):

    '''
    Generic class for getting data and setting up the prior space
    and likelihood. This is tuned for ISGWB analysis at the moment
    but it should not be difficult to modify for other use cases.
    '''

    def __init__(self,  params, inj):

        # set up the LISAdata class
        LISAdata.__init__(self, params, inj)

        # Make noise spectra
        self.which_noise_spectrum()
        self.which_astro_signal()

        # Generate or get mldc data
        if self.params['mldc']:
            self.read_mldc_data()
        else:
            self.makedata()

        # Set up the Bayes class
        Bayes.__init__(self)

        # Figure out which response function to use for recoveries
        self.which_response()

        # Make some simple diagnostic plots to contrast spectra
        self.diag_spectra()

    def makedata(self):

        '''
        Just a wrapper function to use the methods the LISAdata class
        to generate data. Return Frequency domain data.
        '''

        # Generate TDI noise
        times, self.h1, self.h2, self.h3 = self.gen_noise_spectrum()
        delt = times[1] - times[0]

        # Cut to required size
        N = int((self.params['dur'])/delt)
        self.h1, self.h2, self.h3 = self.h1[0:N], self.h2[0:N], self.h3[0:N]

        # Generate TDI isotropic signal
        if self.inj['doInj']:

            h1_gw, h2_gw, h3_gw, times = self.add_sgwb_data()

            h1_gw, h2_gw, h3_gw = h1_gw[0:N], h2_gw[0:N], h3_gw[0:N]

            # Add gravitational-wave time series to noise time-series
            self.h1 = self.h1 + h1_gw
            self.h2 = self.h2 + h2_gw
            self.h3 = self.h3 + h3_gw

        self.timearray = times[0:N]
        if delt != (times[1] - times[0]):
            raise ValueError('The noise and signal arrays are at different sampling frequencies!')

        # Desample if we increased the sample rate for time-shifts.
        if self.params['fs'] != 1.0/delt:
            self.params['fs'] = 1.0/delt

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = self.tser2fser(self.h1, self.h2, self.h3, self.timearray)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)

    def read_mldc_data(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to
        read data. Return frequency domain data. Since this was used
        primarily for the MLDC, this assumes that the data is doppler
        tracking and converts to strain data.
        '''

        h1, h2, h3, self.timearray = self.read_data()

        # Calculate other tdi combinations if necessary.
        if self.params['tdi_lev'] == 'aet':
            h1 = (1.0/3.0)*(2*h1 - h2 - h3)
            h2 = (1.0/np.sqrt(3.0))*(h3 - h2)
            h3 = (1.0/3.0)*(h1 + h2 + h3)

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = self.tser2fser(h1, h2, h3, self.timearray)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)

        # Convert doppler data to strain if readfile datatype is doppler.
        if self.params['datatype'] == 'doppler':

            # This is needed to convert from doppler data to strain data.
            self.r1, self.r2, self.r3 = self.r1/(4*self.f0.reshape(self.f0.size, 1)), self.r2/(4*self.f0.reshape(self.f0.size, 1)), self.r3/(4*self.f0.reshape(self.f0.size, 1))

        elif self.params['datatype'] == 'strain':
            pass


    def which_noise_spectrum(self):

        # Figure out which instrumental noise spectra to use
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

        # Calculate reponse function to use for analysis
        if (self.params['modeltype'] == 'isgwb' or self.params['modeltype'] == 'isgwb_only') and self.params['tdi_lev']=='aet':
            self.response_mat = self.isgwb_aet_response(self.f0, self.tsegmid)

        elif (self.params['modeltype'] == 'isgwb' or self.params['modeltype'] == 'isgwb_only') and self.params['tdi_lev']=='xyz':
            self.response_mat = self.isgwb_xyz_response(self.f0, self.tsegmid)

        elif (self.params['modeltype'] == 'isgwb' or self.params['modeltype'] == 'isgwb_only') and self.params['tdi_lev']=='michelson':
            self.response_mat = self.isgwb_mich_response(self.f0, self.tsegmid)

        elif self.params['modeltype']=='sph_sgwb' and self.params['tdi_lev']=='michelson':
            self.response_mat = self.asgwb_mich_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='sph_sgwb' and self.params['tdi_lev']=='xyz':
            self.response_mat = self.asgwb_xyz_response(self.f0, self.tsegmid)
        elif self.params['modeltype']=='sph_sgwb' and self.params['tdi_lev']=='aet':
            self.response_mat = self.asgwb_aet_response(self.f0, self.tsegmid)

        elif self.params['modeltype'] == 'noise_only':
            print('Noise only model chosen ...')
        else:
            raise ValueError('Unknown recovery model selected')

    def which_astro_signal(self):

        # Figure out which antenna patterns to use
        if self.inj['injtype'] == 'isgwb' and self.params['tdi_lev']=='aet':
            self.add_astro_signal = self.isgwb_aet_response
        elif self.inj['injtype'] == 'isgwb' and self.params['tdi_lev']=='xyz':
            self.add_astro_signal = self.isgwb_xyz_response
        elif self.inj['injtype'] == 'isgwb' and self.params['tdi_lev']=='michelson':
            self.add_astro_signal = self.isgwb_mich_response
        elif self.inj['injtype']=='sph_sgwb' and self.params['tdi_lev']=='michelson':
            self.add_astro_signal = self.asgwb_mich_response
        elif self.inj['injtype']=='sph_sgwb' and self.params['tdi_lev']=='aet':
            self.add_astro_signal = self.asgwb_aet_response
        elif self.inj['injtype']=='sph_sgwb' and self.params['tdi_lev']=='xyz':
            self.add_astro_signal = self.asgwb_xyz_response
        else:
           raise ValueError('Unknown recovery model selected')



    def diag_spectra(self):

        '''
        A function to do simple diagnostics. Plot the expected spectra and data.
        '''

        import scipy.signal as sg

        # ------------ Calculate PSD ------------------

        # Number of segmants

        Nperseg=int(self.params['fs']*self.params['seglen'])

        # PSD from the FFTs
        data_PSD1, data_PSD2, data_PSD3  = np.mean(np.abs(self.r1)**2, axis=1), np.mean(np.abs(self.r2)**2, axis=1), np.mean(np.abs(self.r3)**2, axis=1)

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
        data_PSD1,data_PSD2, data_PSD3 = data_PSD1[idx], data_PSD2[idx], data_PSD3[idx]

        # The last two elements are the position and the acceleration noise levels.
        Np, Na = 10**self.inj['log_Np'], 10**self.inj['log_Na']

        # Modelled Noise PSD
        C_noise = self.instr_noise_spectrum(self.fdata,self.f0, Np, Na)

        # Extract noise auto-power
        S1, S2, S3 = C_noise[0, 0, :], C_noise[1, 1, :], C_noise[2, 2, :]

        if self.params['modeltype'] != 'noise_only':

            if self.params['modeltype'] == 'sph_sgwb':
                alms_inj = self.blm_2_alm(self.inj['blms'])

                # normalize
                alms_inj = alms_inj/(alms_inj[0] * np.sqrt(4*np.pi))

                summ_response_mat = np.sum(self.response_mat*alms_inj[None, None, None, None, :], axis=-1)
                # extra auto-power GW responses
                R1 = np.real(summ_response_mat[0, 0, :, 0])
                R2 = np.real(summ_response_mat[1, 1, :, 0])
                R3 = np.real(summ_response_mat[2, 2, :, 0])

            else:
                # extra auto-power GW responses
                R1 = np.real(self.response_mat[0, 0, :, 0])
                R2 = np.real(self.response_mat[1, 1, :, 0])
                R3 = np.real(self.response_mat[2, 2, :, 0])

            # SGWB signal levels of the mldc data
            Omega0, alpha = 10**self.inj['ln_omega0'], self.inj['alpha']

            # Hubble constant
            H0 = 2.2*10**(-18)

            # Calculate astrophysical power law noise
            Omegaf = Omega0*(self.fdata/25)**alpha

            # Power spectra of the SGWB
            Sgw = (3.0*(H0**2)*Omegaf)/(4*np.pi*np.pi*self.fdata**3)

            # Spectrum of the SGWB signal convoluted with the detector response tensor.
            S1_gw, S2_gw, S3_gw = Sgw*R1, Sgw*R2, Sgw*R3

            # The total noise spectra is the sum of the instrumental + astrophysical
            S1, S2, S3 = S1 + S1_gw, S2 + S2_gw, S3 + S3_gw

            plt.loglog(self.fdata, S1_gw, label='gw required')


        plt.loglog(self.fdata, S3, label='required')
        plt.loglog(psdfreqs, data_PSD3,label='PSD of the data series', alpha=0.6)
        plt.xlabel('f in Hz')
        plt.ylabel('Power Spectrum ')
        plt.legend()
        plt.ylim([1e-44, 5e-40])
        plt.grid()
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])


        plt.savefig(self.params['out_dir'] + '/diag_psd.png', dpi=200)
        print('Diagnostic spectra plot made in ' + self.params['out_dir'] + '/diag_psd.png')
        plt.close()


        # cross-power diag plots. We will only do 12. IF TDI=XYZ this is S_XY and if TDI=AET
        # this will be S_AE

        ii, jj = 2, 0

        if self.params['modeltype'] == 'noise_only':
            Sx = C_noise[ii, jj, :]
        elif self.params['modeltype'] == 'sph_sgwb':
            Sx = C_noise[ii, jj, :] + Sgw*summ_response_mat[ii, jj, :, 0]
        else:
            Sx = C_noise[ii, jj, :] + Sgw*self.response_mat[ii, jj, :, 0]

        CSDx = np.mean(np.conj(self.rbar[:, :, ii]) * self.rbar[:, :, jj], axis=1)

        plt.subplot(2, 1, 1)
        plt.loglog(self.fdata, np.abs(np.real(Sx)), label='Re(Required ' + str(ii+1) + str(jj+1) + ')' )
        plt.loglog(psdfreqs, np.abs(np.real(CSDx)) ,label='Re(CSD' + str(ii+1) + str(jj+1) + ')', alpha=0.6)
        plt.xlabel('f in Hz')
        plt.ylabel('Power in 1/Hz')
        plt.legend()
        plt.ylim([1e-44, 5e-40])
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.loglog(self.fdata, np.abs(np.imag(Sx)), label='Im(Required ' + str(ii+1) + str(jj+1) + ')' )
        plt.loglog(psdfreqs, np.abs(np.imag(CSDx)) ,label='Im(CSD' + str(ii+1) + str(jj+1) + ')', alpha=0.6)
        plt.xlabel('f in Hz')
        plt.ylabel(' Power in 1/Hz')
        plt.legend()
        plt.xlim(0.5*self.params['fmin'], 2*self.params['fmax'])
        plt.ylim([1e-44, 5e-40])
        plt.grid()

        plt.savefig(self.params['out_dir'] + '/diag_csd_' + str(ii+1) + str(jj+1) + '.png', dpi=200)
        print('Diagnostic spectra plot made in ' + self.params['out_dir'] + '/diag_csd_' + str(ii+1) + str(jj+1) + '.png')
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

    config = configparser.ConfigParser()
    config.read(paramsfile)

    # Params Dict
    params['fmin']     = float(config.get("params", "fmin"))
    params['fmax']     = float(config.get("params", "fmax"))
    params['dur']      = float(config.get("params", "duration"))
    params['seglen']   = float(config.get("params", "seglen"))
    params['fs']       = float(config.get("params", "fs"))
    params['Shfile']   = config.get("params", "Shfile")
    params['mldc'] = int(config.get("params", "mldc"))
    params['datatype'] = str(config.get("params", "datatype"))
    params['loadResponse'] = int(config.get("params", "loadResponse"))
    params['loadCustom'] = int(config.get("params", "loadCustom"))
    params['responsefile1']  = str(config.get("params", "responsefile1"))
    params['responsefile2']  = str(config.get("params", "responsefile2"))
    params['responsefile3']  = str(config.get("params", "responsefile3"))
    params['datafile']  = str(config.get("params", "datafile"))
    params['fref'] = float(config.get("params", "fref"))
    params['modeltype'] = str(config.get("params", "modeltype"))
    params['tdi_lev'] = str(config.get("params", "tdi_lev"))
    params['lisa_config'] = str(config.get("params", "lisa_config"))
    params['nside'] = int(config.get("params", "nside"))
    params['lmax'] = int(config.get("params", "lmax"))


    # Injection Dict
    inj['doInj']       = int(config.get("inj", "doInj"))
    inj['injtype']     = str(config.get("inj", "injtype"))
    inj['ln_omega0']   = np.log10(float(config.get("inj", "omega0")))
    inj['alpha']       = float(config.get("inj", "alpha"))
    inj['log_Np']      = np.log10(float(config.get("inj", "Np")))
    inj['log_Na']      = np.log10(float(config.get("inj", "Na")))

    if inj['injtype'] ==  'sph_sgwb':
        blm_vals = config.get("inj", "blms")
        blm_vals = blm_vals.split(',')

        num_blms = int(0.5*(params['lmax'] + 1) * (params['lmax'] + 2))
        blms = np.zeros(num_blms, dtype='complex')

        for ii in range(num_blms):
            blms[ii] = complex(blm_vals[ii])

        inj['blms'] = blms

    # some run parameters
    params['out_dir']            = str(config.get("run_params", "out_dir"))
    params['doPreProc']          = int(config.get("run_params", "doPreProc"))
    params['input_spectrum']     = str(config.get("run_params", "input_spectrum"))
    params['FixSeed']            = str(config.get("run_params", "FixSeed"))
    params['seed']               = int(config.get("run_params", "seed"))
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

    if params['FixSeed']:
        from tools.SetRandomState import SetRandomState as setrs
        seed = params['seed']
        randst = setrs(seed)
    else:
        randst = None


    if params['modeltype']=='isgwb':

        print("Doing an isotropic stochastic analysis...")
        parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$', r'$\alpha$', r'$\log_{10} (\Omega_0)$']
        npar = len(parameters)
        engine = NestedSampler(lisa.isgwb_log_likelihood, lisa.isgwb_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst)

    elif params['modeltype']=='sph_sgwb':

        print("Doing a spherical harmonic stochastic analysis ...")

        # add the basic parameters first
        parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$', r'$\alpha$', r'$\log_{10} (\Omega_0)$']

        # add the blms
        for lval in range(1, params['lmax'] + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    parameters.append(r'$b_{' + str(lval) + str(mval) + '}$' )
                else:
                    parameters.append(r'$|b_{' + str(lval) + str(mval) + '}|$' )
                    parameters.append(r'$\phi_{' + str(lval) + str(mval) + '}$' )

        npar = len(parameters)
        engine = NestedSampler(lisa.sph_log_likelihood, lisa.sph_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst)

    elif params['modeltype']=='noise_only':
        print("Doing an instrumental noise only analysis ...")
        parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
        npar = len(parameters)
        engine = NestedSampler(lisa.instr_log_likelihood,  lisa.instr_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst)

    elif params['modeltype'] =='isgwb_only':
        print("Doing an isgwb signal only analysis ...")
        parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
        npar = len(parameters)
        engine = NestedSampler(lisa.isgwb_only_log_likelihood,  lisa.isgwb_only_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, rstate = randst)

    else:
        raise ValueError('Unknown recovery model selected')

    print("npar = " + str(npar))

    # -------------------- Extract and Plot posteriors ---------------------------
    engine.run_nested(dlogz=0.5,print_progress=True )


    # re-scale weights to have a maximum of one
    res = engine.results
    weights = np.exp(res['logwt'] - res['logz'][-1])
    weights[-1] = 1 - np.sum(weights[0:-1])

    post_samples = resample_equal(res.samples, weights)

    # Pull the evidence and the evidence error
    logz = res['logz']
    logzerr = res['logzerr']

    # Construct filenames based on parameter configuration
    if params['lisa_config']=='stationary':
        configchar = '_s'
    elif params['lisa_config']=='orbiting':
        configchar = '_o'
    else:
        configchar = ''

    if params['FixSeed']:
        seedchar = '_rs'+str(seed)
    else:
        seedchar = ''
    logzname = '/logz'+seedchar+configchar+'.txt'
    logzerrname = '/logzerr'+seedchar+configchar+'.txt'

    # Save posteriors to file
    np.savetxt(params['out_dir'] + "/post_samples.txt",post_samples)
    np.savetxt(params['out_dir'] + logzname,logz)
    np.savetxt(params['out_dir'] + logzerrname,logzerr)

    # Save parameters as a pickle
    outfile = open(params['out_dir'] + '/config.pickle', 'wb')
    pickle.dump(params, outfile)
    pickle.dump(inj, outfile)
    pickle.dump(parameters, outfile)

    print("\n Making posterior Plots ...")
    plotmaker(params, parameters, inj)
    # open_img(params['out_dir'])

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide (only) the params file as an argument')
    else:
        blip(sys.argv[1])
