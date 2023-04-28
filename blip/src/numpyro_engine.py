import numpy as np
#import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist


#from numpy.lib.recfunctions import structured_to_unstructured as s2us
#
##from dynesty.utils import resample_equal
#import dill
##import time
#import shutil, os
#import json


#class numpyro_wrapper():
#    
#    def __init__(self,Model):
#        self.Model = Model
#    
#    def prior_transform(self,theta,theta_trans):
#        return theta_trans
#    
#    def numpyro_model(self):
#        '''
#        Wrapper to translate our unified prior and likelihood to Numpyro-friendly input.
#        
#        Arguments
#        -------------
#        parameters (list of str)    : Model parameter name list
#        prior_transform (function) : prior transform from unit cube
#        log_likelihood (function) : desired log likelihood function, should take in theta parameter vector (we will do live adaptation to numpyro format below)
#        '''
#        
#        with numpyro.plate("Npar",self.Model.Npar):
#            theta = numpyro.sample('theta',dist.Uniform(0,1))
#        
#        theta_trans = self.Model.prior(theta)
#        
#        theta_trans = numpyro.deterministic("theta_transformed", self.prior_transform(theta,theta_trans))
#        
#        numpyro.factor('loglike',log_factor=self.Model.likelihood(theta_trans))
        

#@partial(jax.jit,static_argnums=0)

def numpyro_model(Model):
    '''
    Wrapper to translate our unified prior and likelihood to Numpyro-friendly input.
    
    Arguments
    -------------
    parameters (list of str)    : Model parameter name list
    prior_transform (function) : prior transform from unit cube
    log_likelihood (function) : desired log likelihood function, should take in theta parameter vector (we will do live adaptation to numpyro format below)
    '''
    
    with numpyro.plate("Npar",Model.Npar):
        theta = numpyro.sample('theta',dist.Uniform(0,1))
    
    theta_trans = numpyro.deterministic("theta_transformed", Model.prior(theta))
#    import pdb; pdb.set_trace()
    numpyro.factor('loglike',log_factor=Model.likelihood(theta_trans))


class numpyro_engine():

    '''
    Class for interfacing with numpyro sampler. This method also contains the
    priors definition for all models written to work with the nessai sampler.
    '''
    
    @classmethod
    def define_engine(cls, lisaobj, Nburn, Nsamples, Nthreads, seed):
        
        ## multithreading setup, will need to tweak if we port to GPU
        numpyro.set_host_device_count(Nthreads)
        
        if seed is not None:
            rng_key = jax.random.PRNGKey(seed)
        else:
            raise TypeError("Numpyro sampler requires a defined seed.")
        
#        wrapped_model = numpyro_wrapper(lisaobj.Model)
        
        kernel = NUTS(numpyro_model)
        
        engine = MCMC(kernel,num_warmup=Nburn,num_samples=Nsamples,num_chains=Nthreads)

        # print npar
        print("Npar = " + str(lisaobj.Model.Npar))

        return engine, lisaobj.Model.parameters, rng_key
    
    @staticmethod
    def run_engine(engine,lisaobj,rng_key):
        
        # -------------------- Run HMC sampler ---------------------------
        engine.run(rng_key,lisaobj.Model)
        
        ## retrive samples and reformat
        post_samples = np.array(engine.get_samples()['theta_transformed']).T
        
        return post_samples

    ## no checkpointing for now
#    def load_engine(params,nlive,nthread,seed,output,checkpoint_interval=None):
#        
#        ## load model and parameters from previous checkpoint
#        resume_file = params['out_dir']+'/checkpoint.pickle'
#        if os.path.isfile(resume_file):
#            print("Loading interrupted analysis from last checkpoint...")
#            with open(resume_file,'rb') as file:
#                flow_model, parameters = dill.load(file)
#        else:
#            raise ValueError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
#        
#        # create multiprocessing pool
#        if nthread > 1:
#            pool_size = nthread
#        else:
#            pool_size = None
#
#        ## use nessai's internal checkpointing to reload the engine
#        engine = FlowSampler(flow_model,nlive=nlive,output=output,seed=seed,stopping=0.1,n_pool=pool_size,checkpoint_interval=checkpoint_interval,
#                             resume=True)#,resume_file=params['out_dir']+'/nessai_output/nested_sampler_resume.pkl')
#        
#        return engine, parameters, flow_model
#    
#    @staticmethod
#    def run_engine_with_checkpointing(engine,parameters,model,output,blip_checkpoint_file):
#        
#        ## nessai has very nice internal checkpointing, so all we need to do is save BLIP's state so we can pass along the model and data upon resuming a run
#        if dill.pickles([model,parameters]):
#            temp_file = blip_checkpoint_file + ".temp"
#            with open(temp_file, "wb") as file:
#                dill.dump([model,parameters], file)
#            shutil.move(temp_file, blip_checkpoint_file)
#        else:
#            print("Warning: Cannot write checkpoint file, job cannot resume if interrupted.")
#        # -------------------- Run nested sampler ---------------------------
#        logger = setup_logger(output=output)
#        engine.run()
#        
#        with open(output+'/result.json', 'r') as file:
#            res = json.load(file)
#        ## samples on the n-D unit cube
#        unit_samples = [np.array(res['posterior_samples'][name]) for name in parameters['all']]
#        post_samples = np.vstack(model.prior_transform(unit_samples)).T
#        
#        logz = res['log_evidence']
#        logzerr = res['log_evidence_error']
#        
#        return post_samples, logz, logzerr
    

    

