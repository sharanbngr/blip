import numpy as np
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import pickle, dill
import os
import shutil


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
    Class for interfacing with numpyro sampler. 
    '''
    
    @classmethod
    def define_engine(cls, lisaobj, Nburn, Nsamples, Nthreads, prog, seed):
        
        ## multithreading setup, will need to tweak if we port to GPU
        numpyro.set_host_device_count(Nthreads)
        
        if seed is not None:
            rng_key = jax.random.PRNGKey(seed)
        else:
            raise TypeError("Numpyro sampler requires a defined seed.")
        
        kernel = NUTS(numpyro_model)
        
        engine = MCMC(kernel,num_warmup=Nburn,num_samples=Nsamples,num_chains=Nthreads,progress_bar=prog)

        # print npar
        print("Npar = " + str(lisaobj.Model.Npar))

        return engine, lisaobj.Model.parameters, rng_key
    
    @staticmethod
    def run_engine(engine,lisaobj,rng_key):
        
        # -------------------- Run HMC sampler ---------------------------
        print("Beginning sampling...")
        engine.run(rng_key,lisaobj.Model)
        print("Sampling complete. Retrieving posterior and plotting results...")
        ## retrive samples and reformat
        post_samples = np.array(engine.get_samples()['theta_transformed']).T
        
        return post_samples
    
    
    def load_engine(resume_file):
        
        ## load model and parameters from previous checkpoint
        if os.path.isfile(resume_file):
            print("Loading interrupted analysis from last checkpoint...")
            with open(resume_file,'rb') as file:
                engine,state,chain = pickle.load(file)
            ## tell numpyro to start from current state
            engine.post_warmup_state = state
            ## grab rng_key for running
            rng_key = engine.post_warmup_state.rng_key
        else:
            raise TypeError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
    
        return engine, rng_key, chain
    
    
    @staticmethod
    def run_engine_with_checkpointing(engine,lisaModel,rng_key,chain,checkpoint_file,Ntotal):

        if chain is None:
            print("Beginning sampling, starting warmup phase...")
            
        
        while True:
            
            engine.run(rng_key,lisaModel)
            
            ## get state, current chain
            state = engine.last_state
            chain_update = engine.get_samples()
            if chain is not None:
                chain_updated = {}
                chain_updated['theta'] = jnp.append(chain['theta'],chain_update['theta'],axis=0)
                chain_updated['theta_transformed'] = [jnp.append(chain['theta_transformed'][i],chain_update['theta_transformed'][i]) for i in range(len(chain_update['theta_transformed']))]
                chain = chain_updated
            else:
                chain = chain_update
            
            ## check to see if we have the desired number of samples yet
            Ncurrent = len(chain['theta_transformed'][0])
            if Ncurrent >= Ntotal:
                break

            print("Checkpointing ({}/{} samples)...".format(Ncurrent,Ntotal))
            ## save
            if dill.pickles([engine,state,chain]):
                temp_file = checkpoint_file + ".temp"
                with open(temp_file, "wb") as file:
                    pickle.dump([engine,state,chain], file)
                shutil.move(temp_file, checkpoint_file)
            else:
                print("Warning: Cannot write checkpoint file, job cannot resume if interrupted.")
            
            ## tell numpyro to start from current state
            engine.post_warmup_state = state
            ## grab rng_key for running
            rng_key = engine.post_warmup_state.rng_key

        ## retrive samples and reformat
        post_samples = np.array(chain['theta_transformed']).T
        
        return post_samples

