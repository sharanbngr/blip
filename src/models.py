import numpy as np

from src.geometry import geometry
from src.sph_geometry import sph_geometry
from src.populations import populations
from src.likelihoods import likelihoods



class submodel(geometry,sph_geometry):
    '''
    Modular class that can represent either an injection or an analysis model. Will have different attributes depending on use case.
    
    Includes all information required to generate an injection or a likelihood/prior.
    
    New models (injection or analysis) should be added here.
    '''
    def __init__(self,params,inj,submodel_name,submodel_args=None,injection=False):
        ## preliminaries
        self.params = params
        self.inj = inj
        
        ###################################################
        ###            BUILD NEW MODELS HERE            ###
        ###################################################
        ## check if custom submodel has been passed
        if submodel_args is not None:
            print("Generating user-provided submodel...")
            ## NB - need to put some checks in here to make sure the dictionary has all the required keys
            print("Provided submodel dictionary is:")
            print(submodel_args)
        else:
            submodel_args = {}
            submodel_args['parameters'] = []
            
            ## built-in submodels
            if submodel_name == 'powerlaw_isgwb':
                submodel_args['spectrum'] = 'powerlaw'
                submodel_args['sky_dist'] = 'isotropic'

            
        
        
        
        
        
        ## assignment of spectrum
        if submodel_args['spectrum'] == 'powerlaw':
            self.parameters = self.parameters + [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            self.omegaf = self.powerlaw_spectrum
        else:
            ValueError("Unsupported spectrum type. Check your spelling or add a new spectrum model!")
        
        ## assignment of response
        if submodel_args['sky_dist'] == 'isotropic':
            if self.params['tdi_lev'] == 'michelson':
                self.response = self.isgwb_mich_response
            elif self.params['tdi_lev'] == 'xyz':
                self.response = self.isgwb_xyz_response
            elif self.params['tdi_lev'] == 'aet':
                self.response = self.isgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
        elif submodel_args['sky_dist'] == 'anisotropic':
            pass
        else:
            raise ValueError("Invalid specification of sky_dist. Can be 'isotropic' or 'anisotropic'.")
        
        if not injection:               
            submodel_args['Npar'] = len(submodel_args['parameters'])
    
    def powerlaw_spectrum(self,fs,log_omega0,alpha):
        return 10**(log_omega0)*(fs/self.params['fref'])**alpha




class Model(likelihoods):
    '''
    Class to house all model attributes in a modular fashion.
    '''
    def __init__(self,params,inj):
        
        pass
    
    def compute_OmegaGW(self,fs):
        pass
    
    def compute_Sgw(self,fs):
        H0 = 2.2*10**(-18)
        Omegaf = self.compute_OmegaGW(fs)
        Sgw = Omegaf*(3/(4*fs**3))*(H0/np.pi)**2
        return Sgw

    
class Injection(geometry,sph_geometry,populations):
    pass

