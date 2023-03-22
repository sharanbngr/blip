import numpy as np
from matplotlib import pyplot as plt

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
    def __init__(self,params,inj,submodel_name,fs,f0,tsegmid,injection=False):
        ## preliminaries
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters
        self.fs = fs
        self.f0= f0
        self.time_dim = tsegmid.size
        self.name = submodel_name
            
        ## handle & return noise case in bespoke fashion, as it is quite different from the signal models
        if submodel_name == 'noise':
            self.parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
            self.Npar = 2
            self.fancyname = "Instrumental Noise Spectrum"
            # Figure out which instrumental noise spectra to use
            if self.params['tdi_lev']=='aet':
                self.instr_noise_spectrum = self.aet_noise_spectrum
#                self.gen_noise_spectrum = self.gen_aet_noise
            elif self.params['tdi_lev']=='xyz':
                self.instr_noise_spectrum = self.xyz_noise_spectrum
#                self.gen_noise_spectrum = self.gen_xyz_noise
            elif self.params['tdi_lev']=='michelson':
                self.instr_noise_spectrum = self.mich_noise_spectrum
#                self.gen_noise_spectrum = self.gen_michelson_noise
            else:
                raise ValueError("Unknown specification of 'tdi_lev'; can be 'michelson', 'xyz', or 'aet'.")
            
            ## prior transform
            self.prior = self.instr_noise_prior
            ## covariance calculation
            self.cov = self.compute_cov_noise
            
            return
        
        else:
            self.parameters = []
            self.spectral_model_name, self.spatial_model_name = submodel_name.split('_')
            
            
        
        ###################################################
        ###            BUILD NEW MODELS HERE            ###
        ###################################################
#        ## check if custom submodel has been passed
#        if submodel_args is not None:
#            print("Generating user-provided submodel...")
#            ## NB - need to put some checks in here to make sure the dictionary has all the required keys
#            print("Provided submodel dictionary is:")
#            print(submodel_args)
#        else:
#            submodel_args = {}
#            submodel_args['parameters'] = []
#            
#            ## built-in submodels
#            if submodel_name == 'powerlaw_isgwb':
#                submodel_args['spectrum'] = 'powerlaw'
#                submodel_args['sky_dist'] = 'isotropic'       
        
        
        ## assignment of spectrum
        if self.spectral_model_name == 'powerlaw':
            self.parameters = self.parameters + [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            self.omegaf = self.powerlaw_spectrum
            if not injection:
                self.spectral_prior = self.powerlaw_prior
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
            self.fancyname = "Isotropic SGWB"
            self.subscript = "_{I}"

            if not injection:
                ## prior transform
                self.prior = self.isotropic_prior
            
        elif self.spatial_model_name == 'sph':
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
        submodel_names = params['model'].split('+')
        
        ## initialize submodels
        self.submodels = []
        self.Npar = 0
        self.parameters = {}
        all_parameters = []
        for submodel_name in submodel_names:
            sm = submodel(params,inj,submodel_name,fs,f0,tsegmid)
            self.submodels.append(sm)
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
        
        for sm in self.submodels:
            theta += sm.prior(unit_theta[start_idx:(start_idx+sm.Npar)])
            start_idx += sm.Npar
        
        if len(theta) != len(unit_theta):
            raise ValueError("Input theta does not have same length as output theta, something has gone wrong!")
        
        return theta
    
    
    def likelihood(self,theta):
        
        ## need to cleverly unpack the priors
        
        ## then need to compute each submodel's contribution to the covariance matrix
        
        start_idx = 0
        for i, sm in enumerate(self.submodels):
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
    
    
    
    
    
    
    
#    def plot_injection_spectra(self,fs,save=True):
#        
#        plt.figure()
#        
#        
#        pass
    pass




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