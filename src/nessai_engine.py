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
        
        ## determine parameters
        if params['modeltype'] !='isgwb_only':
            noise_parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
        else:
            noise_parameters = []
        if params['modeltype'] !='noise_only':
            if params['spectrum_model']=='powerlaw':
                signal_parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            elif params['spectrum_model']=='broken_powerlaw':
                signal_parameters = [r'$\log_{10} (A_1)$',r'$\alpha_1$',r'$\log_{10} (A_2)$']
            elif params['spectrum_model']=='broken_powerlaw_2':
                signal_parameters = [r'$\log_{10} (\Omega_0)$',r'$\alpha_1$',r'$\alpha_2$',r'$\log_{10} (f_{\mathrm{break}})$']
            elif params['spectrum_model']=='truncated_broken_powerlaw':
                signal_parameters = [r'$\log_{10} (\Omega_0)$',r'$\alpha_1$',r'$\alpha_2$',r'$\log_{10} (f_{\mathrm{break}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            elif params['spectrum_model']=='truncated_powerlaw':
                signal_parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$',r'$\log_{10} (f_{\mathrm{break}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            elif params['spectrum_model']=='free_broken_powerlaw':
                signal_parameters = [r'$\log_{10} (A_1)$',r'$\alpha_1$',r'$\log_{10} (A_2)$',r'$\alpha_2$']
            else:
                raise ValueError("Unknown specification of spectral model. Available options: powerlaw, broken_powerlaw, and free_broken_powerlaw.")
        else:
            signal_parameters = []
        
        ## build config dict for settings shared across all models
        sampler_config = dict(nlive=nlive,
                           output=output,
                           seed=seed,
                           stopping=0.1,
                           n_pool=pool_size,
                           checkpoint_interval=checkpoint_interval,
                           reset_flow=16)
        
        # create the nested sampler objects      
        if params['modeltype']=='isgwb':

            print("Doing an isotropic stochastic analysis...")
            all_parameters = noise_parameters + signal_parameters
            parameters = {'noise':noise_parameters,'signal':signal_parameters,'blm':[],'all':all_parameters}
            npar = len(all_parameters)
            
            if params['spectrum_model']=='powerlaw':
                model = nessai_model(all_parameters,lisaobj.isgwb_pl_log_likelihood,cls.isgwb_pl_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#                
#                engine = NestedSampler(lisaobj.isgwb_pl_log_likelihood, cls.isgwb_pl_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='broken_powerlaw':
                model = nessai_model(all_parameters,lisaobj.isgwb_bpl_log_likelihood,cls.isgwb_bpl_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#                engine = NestedSampler(lisaobj.isgwb_bpl_log_likelihood, cls.isgwb_bpl_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='broken_powerlaw_2':
                model = nessai_model(all_parameters,lisaobj.isgwb_bpl2_log_likelihood,cls.isgwb_bpl2_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
            elif params['spectrum_model']=='free_broken_powerlaw':
                model = nessai_model(all_parameters,lisaobj.isgwb_fbpl_log_likelihood,cls.isgwb_fbpl_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#                engine = NestedSampler(lisaobj.isgwb_fbpl_log_likelihood, cls.isgwb_fbpl_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='truncated_broken_powerlaw':
                model = nessai_model(all_parameters,lisaobj.isgwb_tbpl_log_likelihood,cls.isgwb_tbpl_prior)
            elif params['spectrum_model']=='truncated_powerlaw':
                model = nessai_model(all_parameters,lisaobj.isgwb_tpl_log_likelihood,cls.isgwb_tpl_prior)
            else:
                raise ValueError("Unknown specification of spectral model. Available options: powerlaw, broken_powerlaw, and free_broken_powerlaw.")
            

        elif params['modeltype']=='sph_sgwb':

            print("Doing a spherical harmonic stochastic analysis ...")

            # add the basic parameters first
            blm_parameters = []
            # add the blms
            for lval in range(1, params['lmax'] + 1):
                for mval in range(lval + 1):

                    if mval == 0:
                        blm_parameters.append(r'$b_{' + str(lval) + str(mval) + '}$' )
                    else:
                        blm_parameters.append(r'$|b_{' + str(lval) + str(mval) + '}|$' )
                        blm_parameters.append(r'$\phi_{' + str(lval) + str(mval) + '}$' )

            all_parameters = noise_parameters + signal_parameters + blm_parameters
            parameters = {'noise':noise_parameters,'signal':signal_parameters,'blm':blm_parameters,'all':all_parameters}
            npar = len(all_parameters)
            
            ## manual neuron configuration
            ## theory behind this # of neurons is that the sph. harm. distributions are generally more complicated than the relatively simple spectral parameters
            ## nessai default is 2*npar neurons; for now allow for a flat 32 neurons (we will probably want to update to a more refined approach later)
#            n_neurons = min(2*npar,32)
            n_neurons = npar + 3*len(blm_parameters)
            flow_config = {'model_config':dict(n_neurons=n_neurons)}
            sampler_config['flow_config'] = flow_config
            
            if params['spectrum_model']=='powerlaw':
                model = nessai_model(all_parameters,lisaobj.sph_pl_log_likelihood,cls.sph_pl_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#                engine = NestedSampler(lisaobj.sph_pl_log_likelihood, cls.sph_pl_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='broken_powerlaw':
                model = nessai_model(all_parameters,lisaobj.sph_bpl_log_likelihood,cls.sph_bpl_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#                engine = NestedSampler(lisaobj.sph_bpl_log_likelihood, cls.sph_bpl_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='broken_powerlaw_2':
                model = nessai_model(all_parameters,lisaobj.sph_bpl2_log_likelihood,cls.sph_bpl2_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
            elif params['spectrum_model']=='free_broken_powerlaw':
                model = nessai_model(all_parameters,lisaobj.sph_fbpl_log_likelihood,cls.sph_fbpl_prior)
#                engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#                engine = NestedSampler(lisaobj.sph_fbpl_log_likelihood, cls.sph_fbpl_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='truncated_broken_powerlaw':
                model = nessai_model(all_parameters,lisaobj.sph_tbpl_log_likelihood,cls.sph_tbpl_prior)
            elif params['spectrum_model']=='truncated_powerlaw':
                model = nessai_model(all_parameters,lisaobj.sph_tpl_log_likelihood,cls.sph_tpl_prior)
            else:
                raise ValueError("Unknown specification of spectral model. Available options: powerlaw, broken_powerlaw, and free_broken_powerlaw.")

        elif params['modeltype']=='noise_only':

            print("Doing an instrumental noise only analysis ...")
            noise_parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
            parameters = {'noise':noise_parameters,'signal':[],'blm':[],'all':noise_parameters}
            npar = len(noise_parameters)
            
            model = nessai_model(all_parameters,lisaobj.instr_log_likelihood,cls.instr_prior)
#            engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#            engine = NestedSampler(lisaobj.instr_log_likelihood,  cls.instr_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size,  rstate = randst)

        elif params['modeltype'] =='isgwb_only':

            print("Doing an isgwb signal only analysis ...")
            signal_parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            parameters = {'noise':[],'signal':signal_parameters,'blm':[],'all':signal_parameters}
            npar = len(signal_parameters)

            model = nessai_model(all_parameters,lisaobj.isgwb_only_log_likelihood,cls.isgwb_only_prior)
#            engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval)
#            engine = NestedSampler(lisaobj.isgwb_only_log_likelihood, cls.isgwb_only_prior,\
#                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size,  rstate = randst)

        else:
            raise ValueError('Unknown recovery model selected')

        ## config and model in hand, build the engine
        engine = FlowSampler(model,**sampler_config)
        
        # print npar
        print("npar = " + str(npar))

        return engine, parameters, model
    
#    def load_engine(params,randst,pool):
#        ## load engine from previous checkpoint
#        ## nqueue is set to a negative number to trigger the queue to be refilled before the first iteration.
#        ## randomstate cannot be saved, so we need to set that as well
#        resume_file = params['out_dir']+'/checkpoint.pickle'
#        if os.path.isfile(resume_file):
#            print("Loading interrupted analysis from last checkpoint...")
#            with open(resume_file,'rb') as file:
#                engine, parameters = dill.load(file)
#                if engine.added_live:
#                    engine._remove_live_points()
#                engine.nqueue = -1
#                engine.rstate = randst
#                if pool is not None:
#                    engine.pool = pool
#                    engine.loglikelihood.pool = pool
#                    engine.M = engine.pool.map
#                else:
#                    engine.pool = None
#                    engine.loglikelihood.pool = None
#                    engine.M = map
#        else:
#            raise TypeError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
        
        return engine, parameters
    
    def load_engine(params,nlive,nthread,seed,output,checkpoint_interval=None):
        
        ## load model and parameters from previous checkpoint
        resume_file = params['out_dir']+'/checkpoint.pickle'
        if os.path.isfile(resume_file):
            print("Loading interrupted analysis from last checkpoint...")
            with open(resume_file,'rb') as file:
                model, parameters = dill.load(file)
        else:
            raise ValueError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
        
        # create multiprocessing pool
        if nthread > 1:
            pool_size = nthread
        else:
#            if pool is not None:
#                print("Warning: Nthread=1 but pool has been defined. This shouldn't happen...")
#            pool = None
            pool_size = None

        ## use nessai's internal checkpointing to reload the engine
        engine = FlowSampler(model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval,
                             resume=True)#,resume_file=params['out_dir']+'/nessai_output/nested_sampler_resume.pkl')
        
        return engine, parameters, model
    
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
    
#    @staticmethod
#    def run_engine_with_checkpointing(engine,parameters,interval,checkpoint_file,step=1000):
#
#       # -------------------- Run nested sampler ---------------------------
#        pool = engine.pool
#        old_ncall = engine.ncall
#        start_time = time.time()
#        while True:
#            engine.run_nested(dlogz=0.5,maxiter=step,print_progress=True)
#            if engine.ncall == old_ncall:
#                break
#            old_ncall = engine.ncall
#            if os.path.isfile(checkpoint_file):
#                last_checkpoint_s = time.time() - os.path.getmtime(checkpoint_file)
#            else:
#                last_checkpoint_s = time.time() - start_time
#            if last_checkpoint_s > interval:
#                print("Checkpointing...")
#                ## pause the pool for saving
#                if engine.pool is not None:
#                    engine.pool = None
#                    engine.M = map
#                ## save
#                if dill.pickles([engine,parameters]):
#                    temp_file = checkpoint_file + ".temp"
#                    with open(temp_file, "wb") as file:
#                        dill.dump([engine,parameters], file)
#                    shutil.move(temp_file, checkpoint_file)
#                else:
#                    print("Warning: Cannot write checkpoint file, job cannot resume if interrupted.")
#                ## restart pool if needed
#                if pool is not None:
#                    engine.pool = pool
#                    engine.M = engine.pool.map
#                ## removes live points lumped in with dead points when previous sampling call concluded
#                ## unsure why dynesty requires this, but it seems to be deeply important
#                if engine.added_live:
#                    engine._remove_live_points()
#        # re-scale weights to have a maximum of one
#        res = engine.results
#        weights = np.exp(res['logwt'] - res['logz'][-1])
#        weights[-1] = 1 - np.sum(weights[0:-1])
#
#        post_samples = resample_equal(res.samples, weights)
#
#        # Pull the evidence and the evidence error
#        logz = res['logz']
#        logzerr = res['logzerr']
#
#
#        return post_samples, logz, logzerr
    
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

#
#       # -------------------- Run nested sampler ---------------------------
#       engine.run()
#       
##        engine.run_nested(dlogz=0.5,print_progress=True )
##        
##        # re-scale weights to have a maximum of one
##        res = engine.results
##        weights = np.exp(res['logwt'] - res['logz'][-1])
##        weights[-1] = 1 - np.sum(weights[0:-1])
##
##        post_samples = resample_equal(res.samples, weights)
#
#        # Pull the evidence and the evidence error
#        # nessai writes all this and doesn't hold it in memory, so load the json:
#        with open(output+'/result.json', 'r') as file:
#            res = json.load(file)
#        post_samples = np.vstack([result['posterior_samples'][name] for name in parameters]).T
#        logz = res['log_evidence']
#        logzerr = res['log_evidence_error']
#
#
#        return post_samples, logz, logzerr

    @staticmethod
    def instr_prior(theta):


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

        return (log_Np, log_Na)

    @staticmethod
    def isgwb_pl_prior(theta):


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
        log_Np, log_Na, alpha, log_omega0 = theta

        # Transform to actual priors
        alpha       =  10*alpha-5
        log_omega0  = -10*log_omega0 - 4
        log_Np      = -5*log_Np - 39
        log_Na      = -5*log_Na - 46

        return (log_Np, log_Na, alpha, log_omega0)

    
    @staticmethod
    def isgwb_bpl_prior(theta):


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
        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        ## first powerlaw
        log_A1 = -30*theta[2] - 5
        alpha_1 = 14*theta[3] - 10
        ## second powerlaw
        log_A2 = -30*theta[4] - 5

        return [log_Np, log_Na, log_A1, alpha_1, log_A2]
    
    @staticmethod
    def isgwb_bpl2_prior(theta):


        '''
        Prior function for an isotropic stochastic backgound analysis for the 2nd broken power law model

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
        # Transform to actual priors
        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        log_omega0 = -10*theta[2] - 4
        alpha_1 = 10*theta[3] - 4
        alpha_2 = 40*theta[4]
        log_fbreak = -2*theta[5] - 2
        

        return [log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak]
    
    @staticmethod
    def isgwb_tbpl_prior(theta):


        '''
        Prior function for an isotropic stochastic backgound analysis for a truncated broken power law model

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
        # Transform to actual priors
        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        log_omega0 = -10*theta[2] - 4
        alpha_1 = 10*theta[3] - 4
        alpha_2 = 40*theta[4]
        log_fbreak = -2*theta[5] - 2
        log_fscale = -2*theta[6] - 2
        

        return [log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak, log_fscale]

    @staticmethod
    def isgwb_tpl_prior(theta):


        '''
        Prior function for an isotropic stochastic backgound analysis for a truncated power law model

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
        # Transform to actual priors
        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        log_omega0 = -10*theta[2] - 4
        alpha = 10*theta[3] - 5
        log_fbreak = -2*theta[4] - 2
        log_fscale = -2*theta[5] - 2
        

        return [log_Np, log_Na, log_omega0, alpha, log_fbreak, log_fscale]
        
    @staticmethod
    def isgwb_fbpl_prior(theta):


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


        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46
       
        ## The rest are the spectral model priors
        ## first powerlaw
        log_A1 = -30*theta[2] - 5
        alpha_1 = 34*theta[3] - 20
        ## second powerlaw
        log_A2 = -30*theta[4] - 5
        alpha_2 = 34*theta[5] - 20
        
        return [log_Np, log_Na, log_A1, alpha_1, log_A2, alpha_2]

    @staticmethod
    def sph_pl_prior(theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis with a power law spectral mdoel

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''

        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        # Prior on alpha, and omega_0
        alpha = 8*theta[2] - 4
        log_omega0  = -6*theta[3] - 5

        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[4:]) + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 4

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:

                    # prior on real and imaginary parts
                    # blm_theta.append(6*theta[cnt] - 3)
                    # blm_theta.append(6*theta[cnt + 1] - 3)

                    ## prior on amplitude, phase
                    blm_theta.append(3* theta[cnt])
                    blm_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2

        # rm these three lines later.
        # blm_theta.append(theta[4])
        # blm_theta.append(2*np.pi*theta[5] - np.pi)

        theta = [log_Np, log_Na, alpha, log_omega0] + blm_theta

        return theta

    @staticmethod
    def sph_bpl_prior(theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis with a power law spectral mdoel

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''

        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        ## first powerlaw
        log_A1 = -30*theta[2] - 5
        alpha_1 = 14*theta[3] - 10
        ## second powerlaw
        log_A2 = -30*theta[4] - 5

        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[5:]) + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 5

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    blm_theta.append(3* theta[cnt])
                    blm_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2


        theta = [log_Np, log_Na, log_A1, alpha_1, log_A2] + blm_theta

        return theta

    @staticmethod
    def sph_bpl2_prior(theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis with an updated broken power law spectral mdoel

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        log_omega0 = -10*theta[2] - 4
        alpha_1 = 10*theta[3] - 4
        alpha_2 = 40*theta[4]
        log_fbreak = -2*theta[5] - 2

        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[6:]) + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 6

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    blm_theta.append(3* theta[cnt])
                    blm_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2


        theta = [log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak] + blm_theta

        return theta

    @staticmethod
    def sph_tbpl_prior(theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis with a truncated broken power law spectral mdoel

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        log_omega0 = -10*theta[2] - 4
        alpha_1 = 10*theta[3] - 4
        alpha_2 = 40*theta[4]
        log_fbreak = -2*theta[5] - 2
        log_fscale = -2*theta[6] - 2
        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[7:]) + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 6

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    blm_theta.append(3* theta[cnt])
                    blm_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2

        theta = [log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak, log_fscale] + blm_theta

        return theta

    @staticmethod
    def sph_tpl_prior(theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis with a truncated power law spectral mdoel

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        log_omega0 = -10*theta[2] - 4
        alpha = 10*theta[3] - 5
        log_fbreak = -2*theta[4] - 2
        log_fscale = -2*theta[5] - 2
        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[6:]) + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 6

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    blm_theta.append(3* theta[cnt])
                    blm_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2

        theta = [log_Np, log_Na, log_omega0, alpha, log_fbreak, log_fscale] + blm_theta

        return theta


    @staticmethod
    def sph_fbpl_prior(theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis with a power law spectral mdoel

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''

        # The first two are the priors on the position and acc noise terms.
        log_Np = -4*theta[0] - 39
        log_Na = -4*theta[1] - 46

        ## The rest are the spectral model priors
        ## first powerlaw
        log_A1 = -30*theta[2] - 5
        alpha_1 = 34*theta[3] - 20
        ## second powerlaw
        log_A2 = -30*theta[4] - 5
        alpha_2 = 34*theta[5] - 20

        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[6:]) + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 6

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    blm_theta.append(3* theta[cnt])
                    blm_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2


        theta = [log_Np, log_Na, log_A1, alpha_1, log_A2, alpha_2] + blm_theta

        return theta
    
    
    @staticmethod
    def isgwb_only_prior(self, theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref

        '''

        # Unpack: Theta is defined in the unit cube
        alpha, log_omega0  = theta

        # Transform to actual priors
        alpha       = 10*alpha-5
        log_omega0  = -10*log_omega0 - 4

        return (alpha, log_omega0)










