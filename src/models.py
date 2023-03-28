import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from src.geometry import geometry
from src.sph_geometry import sph_geometry
from src.populations import populations
from src.likelihoods import likelihoods
from src.instrNoise import instrNoise



class submodel(geometry,sph_geometry,instrNoise):
    '''
    Modular class that can represent either an injection or an analysis model. Will have different attributes depending on use case.
    
    Includes all information required to generate an injection or a likelihood/prior.
    
    New models (injection or analysis) should be added here.
    '''
    def __init__(self,params,inj,submodel_name,fs,f0,tsegmid,injection=False,suffix=''):
        ## preliminaries
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters
        self.fs = fs
        self.f0= f0
        self.time_dim = tsegmid.size
        self.name = submodel_name
        if injection:
            self.truevals = {}
            
        ## handle & return noise case in bespoke fashion, as it is quite different from the signal models
        if submodel_name == 'noise':
            self.parameters = [r'$\log_{10} (Np)$'+suffix, r'$\log_{10} (Na)$'+suffix]
            self.Npar = 2
            ## for plotting
            self.fancyname = "Instrumental Noise"
            self.color = 'dimgrey'
            # Figure out which instrumental noise spectra to use
            if self.params['tdi_lev']=='aet':
                self.instr_noise_spectrum = self.aet_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_aet_noise
            elif self.params['tdi_lev']=='xyz':
                self.instr_noise_spectrum = self.xyz_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_xyz_noise
            elif self.params['tdi_lev']=='michelson':
                self.instr_noise_spectrum = self.mich_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_michelson_noise
            else:
                raise ValueError("Unknown specification of 'tdi_lev'; can be 'michelson', 'xyz', or 'aet'.")
            if not injection:
                ## prior transform
                self.prior = self.instr_noise_prior
                ## covariance calculation
                self.cov = self.compute_cov_noise
            else:
                ## truevals
                self.truevals[r'$\log_{10} (Np)$'] = self.inj['log_Np']
                self.truevals[r'$\log_{10} (Na)$'] = self.inj['log_Na']
                ## save the frozen noise spectra
                self.frozen_spectra = self.instr_noise_spectrum(self.fs,self.f0,Np=10**self.inj['log_Np'],Na=10**self.inj['log_Na'])
            
            return
        
        else:
            self.parameters = []
            self.spectral_model_name, self.spatial_model_name = submodel_name.split('_')
            
            
        
        ###################################################
        ###            BUILD NEW MODELS HERE            ###
        ###################################################
        
        ## color dictionary, for plotting. 
#        color_dict = {'powerlaw_isgwb':'darkorange',
#                      }
        
        
        
        
        ## assignment of spectrum
        if self.spectral_model_name == 'powerlaw':
            self.parameters = self.parameters + [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            self.omegaf = self.powerlaw_spectrum
            self.fancyname = "Power Law SGWB"
            if not injection:
                self.spectral_prior = self.powerlaw_prior
            else:
                self.truevals[r'$\alpha$'] = self.inj['alpha']
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.inj['log_omega0']
        else:
            ValueError("Unsupported spectrum type. Check your spelling or add a new spectrum model!")
        
        ## assignment of response and spatial methods
        if self.spatial_model_name == 'isgwb':
            if self.params['tdi_lev'] == 'michelson':
                self.response = self.isgwb_mich_response
            elif self.params['tdi_lev'] == 'xyz':
                self.response = self.isgwb_xyz_response
            elif self.params['tdi_lev'] == 'aet':
                self.response = self.isgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## plotting stuff
            self.fancyname = "Isotropic "+self.fancyname
            self.subscript = "_{I}"
            self.color='darkorange'

            if not injection:
                ## prior transform
                self.prior = self.isotropic_prior
            
        elif self.spatial_model_name == 'sph':
            
#            self.blm_start = ...
            pass
        
        elif self.spatial_model_name == 'hierarchical':
            pass
        else:
            raise ValueError("Invalid specification of spatial model name. Can be 'isgwb', 'asgwb', or 'hierarchical'.")
        
        ## compute response matrix
        self.response_mat = self.response(f0,tsegmid)

        
        
        if not injection:               
            self.Npar = len(self.parameters)
            ## covariance calculation
            self.cov = self.compute_cov_gw
            
        ## add suffix to parameter names and trueval keys, if desired
        ## (we need this in the multi-model or duplicate model case)
        if suffix != '':
            if injection:
                updated_truevals = {parameter+suffix:self.truevals[parameter] for parameter in self.parameters}
                self.truevals = updated_truevals
            updated_parameters = [parameter+suffix for parameter in self.parameters]
            self.parameters = updated_parameters
            
        
        return
    

    #############################
    ##    Spectral Functions   ##
    #############################
    def powerlaw_spectrum(self,fs,alpha,log_omega0):
        return 10**(log_omega0)*(fs/self.params['fref'])**alpha
    
    
    
    
    
    
    def compute_Sgw(self,fs,omegaf_args):
        H0 = 2.2*10**(-18)
        Omegaf = self.omegaf(fs,*omegaf_args)
        Sgw = Omegaf*(3/(4*fs**3))*(H0/np.pi)**2
        return Sgw
    
    #############################
    ##          Priors         ##
    #############################
    def isotropic_prior(self,theta):
        '''
        Isotropic prior transform. Just serves as a wrapper for the spectral prior, as no additional foofaraw is necessary.
        '''
        return self.spectral_prior(theta)
    
    def sph_prior(self,theta):
        '''
        Spherical harmonic anisotropic prior transform. Combines a generic prior function with ther spherical harmonic priors for the desired lmax.
        '''
        
        ## compute the number of spectral components here by subtracting the expected number of blms
        
#        blm_start = ???
#        spectral_theta = self.spectral_prior(theta[:blm_start])
#        sph_theta = ???
#        return spectral_theta+sph_theta
        pass
    
    def hierarchical_prior(self,theta):
        pass
        
        
    def instr_noise_prior(self,theta):


        '''
        Prior function for only instrumental noise

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref, Np and Na

        '''


        # Unpack: Theta is defined in the unit cube
        log_Np, log_Na = theta

        # Transform to actual priors
        log_Np = -5*log_Np - 39
        log_Na = -5*log_Na - 46

        return [log_Np, log_Na]
    
    def powerlaw_prior(self,theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref, Np and Na

        '''


        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0 = theta

        # Transform to actual priors
        alpha       =  10*alpha-5
        log_omega0  = -10*log_omega0 - 4
        

        return [alpha, log_omega0]
    
    #############################
    ## Covariance Calculations ##
    #############################
    def compute_cov_noise(self,theta):
        '''
        Computes the noise covariance for a given draw of log_Np, log_Na
        '''
        ## unpack priors
        log_Np, log_Na = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        ## Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fs,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.time_dim, axis=3)
        
        return cov_noise
    
    def compute_cov_gw(self,theta):
        '''
        Computes the covariance matrix contribution from a generic stochastic GW signal.
        '''
        ## Signal PSD
        Sgw = self.compute_Sgw(self.fs,theta)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat
        
        return cov_sgwb



class Model(likelihoods):
    '''
    Class to house all model attributes in a modular fashion.
    '''
    def __init__(self,params,inj,fs,f0,tsegmid,rmat):
        
        '''
        Model() parses a Model string from the params file. This is of the form of an arbitrary number of "+"-delimited submodel types.
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".
        
        e.g., "noise+powerlaw_isgwb+truncated-powerlaw_sph" defines a model with noise, an isotropic SGWB with a power law spectrum,
            and a (spherical harmonic model for) an anisotropic SGWB with a truncated power law spectrum.
        '''
        
        self.params = params
        
        ## separate into submodels
        self.submodel_names = params['model'].split('+')
        
        ## separate into submodels
        base_component_names = params['model'].split('+')
        
        ## check for and differentiate duplicate injections
        ## this will append 1 (then 2, then 3, etc.) to any duplicate submodel names
        ## we will also generate appropriate variable suffixes to use in plots, etc..
        self.submodel_names = catch_duplicates(base_component_names)
        suffixes = gen_suffixes(base_component_names)
        
        ## initialize submodels
        self.submodels = {}
        self.Npar = 0
        self.parameters = {}
        all_parameters = []
        for submodel_name, suffix in zip(self.submodel_names,suffixes):
            sm = submodel(params,inj,submodel_name,fs,f0,tsegmid,suffix=suffix)
            self.submodels[submodel_name] = sm
            self.Npar += sm.Npar
            self.parameters[submodel_name] = sm.parameters
            all_parameters += sm.parameters
        self.parameters['all'] = all_parameters
        
        ## assign reference to data for use in likelihood
        self.rmat = rmat
        

    

    def prior(self,unit_theta):
        '''
        Function to interatively perform prior draws for each submodel in the proper order
        
        Arguments
        ----------------
        unit_theta (array) : draws from the unit cube
        
        Returns
        ----------------
        theta (list) : transformed prior draws for all submodels in sequence
        '''
        theta = []
        start_idx = 0
        
        for sm_name in self.submodel_names:
            sm = self.submodels[sm_name]
            theta += sm.prior(unit_theta[start_idx:(start_idx+sm.Npar)])
            start_idx += sm.Npar
        
        if len(theta) != len(unit_theta):
            raise ValueError("Input theta does not have same length as output theta, something has gone wrong!")
        
        return theta
    
    
    def likelihood(self,theta):
        
        ## need to cleverly unpack the priors
        
        ## then need to compute each submodel's contribution to the covariance matrix
        
        start_idx = 0
        for i, sm_name in enumerate(self.submodel_names):
            sm = self.submodels[sm_name]
            theta_i = theta[start_idx:(start_idx+sm.Npar)]
            start_idx += sm.Npar
            if i==0:
                cov_mat = sm.cov(theta_i)
            else:
                cov_mat = cov_mat + sm.cov(theta_i)

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike
    
    
    
    def plot_model(self,fs,save=True):
        
        plt.figure()
        
        
        pass
    

    
class Injection(geometry,sph_geometry,populations):
    '''
    Class to house all model attributes in a modular fashion.
    '''
    def __init__(self,params,inj,fs,f0,tsegmid):
        self.params = params
        self.inj = inj
        
        self.frange = fs
        self.f0 = f0
        self.tsegmid = tsegmid
        
        ## separate into components
        base_component_names = inj['injection'].split('+')
        
        ## check for and differentiate duplicate injections
        ## this will append 1 (then 2, then 3, etc.) to any duplicate component names
        ## we will also generate appropriate variable suffixes to use in plots, etc..
        self.component_names = catch_duplicates(base_component_names)
        ## it's useful to have a version of this without the detector noise
        self.sgwb_component_names = [name for name in self.component_names if name!='noise']
        suffixes = gen_suffixes(base_component_names)
                        
        ## initialize components
        self.components = {}
        self.truevals = {}
        for component_name, suffix in zip(self.component_names,suffixes):
            cm = submodel(params,inj,component_name,fs,f0,tsegmid,injection=True,suffix=suffix)
            self.components[component_name] = cm
            self.truevals.update(cm.truevals)
    
        
    
    
    
    def compute_convolved_spectra(self,component_name,fs_new=None,channels='11',return_fs=False,imaginary=False):
        '''
        Wrapper to convolve the frozen response with the frozen injected GW spectra for the desired channels.
        '''
        
        ## split the channel indicators
        c1_idx, c2_idx = int(channels[0]) - 1, int(channels[1]) - 1
        if not imaginary:
            PSD = np.mean(self.components[component_name].frozen_spectra[:,None] * np.real(self.components[component_name].response_mat[c1_idx,c2_idx,:,:]),axis=1)
        else:
            PSD = np.mean(self.components[component_name].frozen_spectra[:,None] * 1j * np.imag(self.components[component_name].response_mat[c1_idx,c2_idx,:,:]),axis=1)
        
        if fs_new is not None:
            PSD_interp = interp1d(self.frange,np.log10(PSD))
            PSD = 10**PSD_interp(fs_new)
            fs = fs_new
        else:
            fs = self.frange
        
        if return_fs:
            return fs, PSD
        else:
            return PSD

    
    def plot_injected_spectra(self,component_name,fs_new=None,ax=None,convolved=False,legend=False,channels='11',return_PSD=False,scale='log',flim=None,**plt_kwargs):
        '''
        Wrapper to plot the injected spectrum component on the specified matplotlib axes (or current axes if unspecified).
        '''
        
        ## set axes
        if ax is None:
            ax = plt.gca()
        
        ## set fmin/max to specified values, or default to the ones in params
        if flim is not None:
            fmin = flim[0]
            fmax = flim[1]
        else:
            fmin = self.params['fmin']
            fmax = self.params['fmax']
        
        ## get frozen injected spectra at original injection frequencies and convolve with detector response if desired
        if convolved:
            if component_name == 'noise':
                raise ValueError("Cannot convolve noise spectra with the detector GW response - this is not physical. (Set convolved=False in the function call!)")
            PSD = self.compute_convolved_spectra(component_name,channels=channels)
        else:
            PSD = self.components[component_name].frozen_spectra
            ## noise will return the 3x3 covariance matrix, need to grab the desired channel cross-/auto-power
            ## generically capture anything that looks like a covariance matrix for future-proofing
            if (len(PSD.shape)==3) and (PSD.shape[0]==PSD.shape[1]==3):
                I, J = int(channels[0]) - 1, int(channels[1]) - 1
                PSD = PSD[I,J,:]
            
        
        ## downsample (or upsample, but why) if desired
        ## do the interpolation in log-space for better low-f fidelity
        if fs_new is not None:
            PSD_interp = interp1d(self.frange,np.log10(PSD))
            PSD = 10**PSD_interp(fs_new)
            fs = fs_new
        else:
            fs = self.frange
        
        filt = (fs>fmin)*(fs<fmax)
        
        if legend:
            label = self.components[component_name].fancyname
            if plt_kwargs is None:
                plt_kwargs = {}
                plt_kwargs['label'] = label
            else:
                if 'label' not in plt_kwargs.keys():
                    plt_kwargs['label'] = label
        
        if scale=='log':
            ax.loglog(fs[filt],PSD[filt],**plt_kwargs)
        elif scale=='linear':
            ax.plot(fs[filt],PSD[filt],**plt_kwargs)
        else:
            raise ValueError("We only support linear and log plots, there is no secret third option!")
        
        
        if return_PSD:
            return PSD
        else:
            return




def catch_duplicates(names):
    '''
    Function to catch duplicate names so we don't overwrite keys while building a Model or Injection
    
    Arguments
    ---------------
    names (list of str) : model or injection submodel names
    
    Returns
    ---------------
    names (list of str) : model or injection submodel names, with duplicates numbered
    '''
    original_names = names.copy()
    duplicate_check = {name:names.count(name) for name in names}
    for key in duplicate_check.keys():
        if duplicate_check[key] > 1:
            cnt = 1
            for i, original_name in enumerate(original_names):
                if original_name == key:
                    names[i] = original_name + '-' + str(cnt)
    
    return names

def gen_suffixes(names):
    '''
    Function to generate appropriate parameter suffixes so repeated parameters are clearly linked to their respective submodel configurations.
    
    Arguments
    ---------------
    names (list of str) : model or injection submodel names
    
    Returns
    ---------------
    suffixes (list of str) : parameter suffixes for each respective model or injection submodel
    '''
    ## grab the spatial designation (or just 'noise' for the noise case)
    end_lst = [name.split('_')[-1] for name in names]
    ## if we just have noise and a lone signal, we don't need to do this.
    if ('noise' in end_lst) and len(end_lst)==2:
        suffixes = ['','']
        return suffixes
    ## set up our building blocks and model counts for iterative numbering
    shorthand = {'noise':{'abbrv':'','count':1},
                 'isgwb':{'abbrv':'I','count':1},
                 'sph':{'abbrv':'A','count':1},
                 'hierarchical':{'abbrv':'H','count':1} }
    
    suffixes = ['  $\mathrm{[' for i in range(len(names))]
    
    ## find duplicates and count them
    dupc = {end:end_lst.count(end) for end in end_lst}
    
    ## generate the suffixes by assigning the abbreviated notation and numbering as necessary
    for i, (end,suff) in enumerate(zip(end_lst,suffixes)):
        if end == 'noise':
            if dupc[end] > 1:
                raise ValueError("Multiple noise injections/models is not supported.")
            else:
                suffixes[i] = ''
        elif dupc[end] == 1:
            suffixes[i] = suff + shorthand[end]['abbrv'] + ']}$'
        else:
            suffixes[i] = suff + shorthand[end]['abbrv'] + '_' + str(shorthand[end]['count']) + ']}$'
            shorthand[end]['count'] += 1

    return suffixes

def bespoke_inv(A):


    """

    compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed

    Credit to Eelco Hoogendoorn at stackexchange for this piece of wizardy. This is > 3 times
    faster than numpy's det and inv methods used in a fully vectorized way as of numpy 1.19.1

    https://stackoverflow.com/questions/21828202/fast-inverse-and-transpose-matrix-in-python

    """


    AI = np.empty_like(A)

    for i in range(3):
        AI[...,i,:] = np.cross(A[...,i-2,:], A[...,i-1,:])

    det = np.einsum('...i,...i->...', AI, A).mean(axis=-1)

    inv_T =  AI / det[...,None,None]

    # inverse by swapping the inverse transpose
    return np.swapaxes(inv_T, -1,-2), det