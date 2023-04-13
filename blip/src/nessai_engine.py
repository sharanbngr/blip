import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2us
from nessai.flowsampler import FlowSampler
from nessai.model import Model
from nessai.utils import setup_logger
#from dynesty.utils import resample_equal
import dill
#import time
import shutil, os
import json

## nessai needs a defined Model
## we will use this as an adaptor from our existing code structure to what nessai expects
class nessai_model(Model):
    
    '''
    Subclass to dynamically generate a nessai Model instance depending on desired signal model
    '''
    
    def __init__(self,parameters,transformed_log_likelihood,prior_transform):
        '''
        Parameters:
        -------------------------
        parameters (list of str) : list of parameter names
        log_likelihood (function) : desired log likelihood function, should take in theta parameter vector (we will do live adaptation to nessai format below)
        prior (function) : prior transform from unit cube
        
        Note: we use "theta" to refer to the unstructured unit cube and "x" to refer to the corresponding nessai structured array
        
        IMPORTANT: in converting between our unit-cube style approach and the nessai structured arrays, we assume that the parameter order as defined in 
                    the parameters variable (which becomes self.names) matches the order in which theta is unpacked in the provided transformed_log_likelihood.
        '''
        ## Names of parameters to sample
        self.names = parameters
        ## prior bounds will all be on (0,1) as we transform from the unit cube
        self.bounds = {}
        for name in parameters:
            self.bounds[name] = (0,1)
        self.transformed_log_likelihood = transformed_log_likelihood
        self.prior_transform = prior_transform
    
    def log_prior(self,x):
        ## check bounds to get 0 or -inf, ensure float type
        log_p = np.log(self.in_bounds(x), dtype="float")
        ## nominally we would have to also do something like
        ## for n in self.names:
        ##    log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        ## but as theta is on (0,1) for all parameters, the above will always evaluate to 0
        return log_p
    def log_likelihood(self,x):
        log_l = np.zeros(x.size)
        theta = self.prior_transform(s2us(x[self.names]).T)
        log_l += self.transformed_log_likelihood(theta)
        return log_l

class nessai_engine():

    '''
    Class for interfacing with nessai sampler. This method also contains the
    priors definition for all models written to work with the nessai sampler.
    '''

    
    
    @classmethod
    def define_engine(cls, lisaobj, params, nlive, nthread, seed, output, checkpoint_interval=None, resume=False):

        # create multiprocessing pool
        if nthread > 1:
            pool_size = nthread
        else:
#            if pool is not None:
#                print("Warning: Nthread=1 but pool has been defined. This shouldn't happen...")
#            pool = None
            pool_size = None
        
        
        
        
#        ## determine parameters
#        if params['modeltype'] !='isgwb_only':
#            noise_parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
#        else:
#            noise_parameters = []
#        if params['modeltype'] !='noise_only':
#            if params['spectrum_model']=='powerlaw':
#                signal_parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
#            elif params['spectrum_model']=='broken_powerlaw':
#                signal_parameters = [r'$\log_{10} (A_1)$',r'$\alpha_1$',r'$\log_{10} (A_2)$']
#            elif params['spectrum_model']=='broken_powerlaw_2':
#                signal_parameters = [r'$\log_{10} (\Omega_0)$',r'$\alpha_1$',r'$\alpha_2$',r'$\log_{10} (f_{\mathrm{break}})$']
#            elif params['spectrum_model']=='truncated_broken_powerlaw':
#                signal_parameters = [r'$\log_{10} (\Omega_0)$',r'$\alpha_1$',r'$\alpha_2$',r'$\log_{10} (f_{\mathrm{break}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
#            elif params['spectrum_model']=='truncated_powerlaw':
#                signal_parameters = [ r'$\log_{10} (\Omega_0)$',r'$\alpha$',r'$\log_{10} (f_{\mathrm{break}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
#            elif params['spectrum_model']=='free_broken_powerlaw':
#                signal_parameters = [r'$\log_{10} (A_1)$',r'$\alpha_1$',r'$\log_{10} (A_2)$',r'$\alpha_2$']
#            else:
#                raise ValueError("Unknown specification of spectral model. Available options: powerlaw, broken_powerlaw, and free_broken_powerlaw.")
#        else:
#            signal_parameters = []
        
        ## build config dict for settings shared across all models
        sampler_config = dict(nlive=nlive,
                           output=output,
                           seed=seed,
                           stopping=0.1,
                           n_pool=pool_size,
                           checkpoint_interval=checkpoint_interval,
                           reset_flow=params['reset_flow'])
        

        ## manual neuron configuration
        ## theory behind this # of neurons is that the sph. harm. distributions are generally more complicated than the relatively simple spectral parameters
        ## nessai default is 2*npar neurons; for now allow for a flat 32 neurons (we will probably want to update to a more refined approach later)
        if params['nessai_neurons'] is not None:
            if params['nessai_neurons']=='scale_lean':
                n_neurons = min(2*lisaobj.Model.Npar,32)
            elif params['nessai_neurons']=='scale_default':
                n_neurons = 2*lisaobj.Model.Npar
            elif params['nessai_neurons']=='scale_greedy':
                n_neurons = lisaobj.Model.Npar + 3*len(lisaobj.Model.parameters['spatial'])
            elif params['nessai_neurons']=='manual':
                n_neurons = params['n_neurons']
        else:
            n_neurons = 2*lisaobj.Model.Npar
        
        
        flow_config = {'model_config':dict(n_neurons=n_neurons)}
        sampler_config['flow_config'] = flow_config
        
        flow_model = nessai_model(lisaobj.Model.parameters['all'],lisaobj.Model.likelihood,lisaobj.Model.prior)
        
        ## config and model in hand, build the engine
        engine = FlowSampler(flow_model,**sampler_config)
        
        # print npar
        print("Npar = " + str(lisaobj.Model.Npar))

        return engine, lisaobj.Model.parameters,flow_model
        

    
    def load_engine(params,nlive,nthread,seed,output,checkpoint_interval=None):
        
        ## load model and parameters from previous checkpoint
        resume_file = params['out_dir']+'/checkpoint.pickle'
        if os.path.isfile(resume_file):
            print("Loading interrupted analysis from last checkpoint...")
            with open(resume_file,'rb') as file:
                flow_model, parameters = dill.load(file)
        else:
            raise ValueError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
        
        # create multiprocessing pool
        if nthread > 1:
            pool_size = nthread
        else:
            pool_size = None

        ## use nessai's internal checkpointing to reload the engine
        engine = FlowSampler(flow_model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval,
                             resume=True)#,resume_file=params['out_dir']+'/nessai_output/nested_sampler_resume.pkl')
        
        return engine, parameters, flow_model
    
    @staticmethod
    def run_engine_with_checkpointing(engine,parameters,model,output,blip_checkpoint_file):
        
        ## nessai has very nice internal checkpointing, so all we need to do is save BLIP's state so we can pass along the model and data upon resuming a run
        if dill.pickles([model,parameters]):
            temp_file = blip_checkpoint_file + ".temp"
            with open(temp_file, "wb") as file:
                dill.dump([model,parameters], file)
            shutil.move(temp_file, blip_checkpoint_file)
        else:
            print("Warning: Cannot write checkpoint file, job cannot resume if interrupted.")
        # -------------------- Run nested sampler ---------------------------
        logger = setup_logger(output=output)
        engine.run()
        
        with open(output+'/result.json', 'r') as file:
            res = json.load(file)
        ## samples on the n-D unit cube
        unit_samples = [np.array(res['posterior_samples'][name]) for name in parameters['all']]
        post_samples = np.vstack(model.prior_transform(unit_samples)).T
        
        logz = res['log_evidence']
        logzerr = res['log_evidence_error']
        
        return post_samples, logz, logzerr
    

    
    @staticmethod
    def run_engine(engine,parameters,model,output):
        
        # -------------------- Run nested sampler ---------------------------
        logger = setup_logger(output=output)
        engine.run()
        
        with open(output+'/result.json', 'r') as file:
            res = json.load(file)
        ## samples on the n-D unit cube
        unit_samples = [np.array(res['posterior_samples'][name]) for name in parameters['all']]
        post_samples = np.vstack(model.prior_transform(unit_samples)).T
        
        logz = res['log_evidence']
        logzerr = res['log_evidence_error']
        
        return post_samples, logz, logzerr
