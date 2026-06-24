import numpy as np
import jax.numpy as jnp
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
#from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.distributions import constraints
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
    
    numpyro.factor('loglike',log_factor=Model.likelihood(theta_trans))


def numpyro_model_sph(Model):
    '''
    Wrapper to translate our unified prior and likelihood to Numpyro-friendly input.
    
    The _sph version here accounts for the fact that the phase parameter prior should be periodic.
    
    Arguments
    -------------
    parameters (list of str)    : Model parameter name list
    prior_transform (function) : prior transform from unit cube
    log_likelihood (function) : desired log likelihood function, should take in theta parameter vector (we will do live adaptation to numpyro format below)
    '''
    Npar_c = len(Model.blm_phase_idx)
    Npar_nc = Model.Npar - Npar_c
    
    with numpyro.plate("Npar_nc",Npar_nc):
        theta_nc = numpyro.sample('theta_nc',dist.Uniform(0,1))
#        theta = numpyro.sample('theta',dist.ImproperUniform(constraints.real,(),()))
    with numpyro.plate("Npar_c",Npar_c):
        theta_c = numpyro.sample('theta_c',dist.ImproperUniform(constraints.real,(),()))
    
    theta_tot = theta_nc
    
    cnt = 0
    for idx in Model.blm_phase_idx:
        theta_tot = jnp.insert(theta_tot,idx,theta_c[cnt])
        cnt += 1
    
    theta = numpyro.deterministic("theta", theta_tot)
    
    theta_trans = numpyro.deterministic("theta_transformed", Model.prior(theta))
    
    numpyro.factor('loglike',log_factor=Model.likelihood(theta_trans))


class numpyro_engine():

    '''
    Class for interfacing with the Numpyro HMC (NUTS) sampler. 
    '''
    
    @classmethod
    def define_engine(cls, lisaobj, Nburn, Nsamples, Nthreads, prog, seed, gpu=False):
        
        ## multithreading setup
        ## default to parallel chains
        chain_method = 'parallel'
        if gpu:
            ## check number of available gpus
            N_GPU = jax.local_device_count(backend='gpu')
            if N_GPU == 1:
                if Nthreads > 1:
                    print("Nthreads = {}, but only one GPU is available. Setting numpyro chain_method to 'vectorized'.".format(Nthreads))
                    print(" WARNING: Vectorized GPU sampling is an experimental feature and is not stable for all BLIP configurations. If you get an XLA GEMM error, this is probably the cause; revert to standard parallelization in such cases.")
                    chain_method = 'vectorized'
            elif N_GPU > 1:
                if Nthreads > N_GPU:
                    print("Nthreads ({}) > N_GPU ({}) but vectorized parallel sampling has not yet been implemented. Setting Nthreads = N_GPU.".format(Nthreads,N_GPU))
                    Nthreads = N_GPU
            else:
                raise ValueError("GPU usage was requested but no GPUs are available!")
        else:
            numpyro.set_host_device_count(Nthreads)
        
        
        if seed is not None:
            rng_key = jax.random.PRNGKey(seed)
        else:
            raise TypeError("Numpyro sampler requires a defined seed.")
        
        ## if there are phase parameters, use the sph wrapper
        if len(lisaobj.Model.blm_phase_idx) > 0:
            kernel = NUTS(numpyro_model_sph)
        ## otherwise use the standard one
        else:
            kernel = NUTS(numpyro_model)
        
        engine = MCMC(kernel,num_warmup=Nburn,num_samples=Nsamples,num_chains=Nthreads,chain_method=chain_method,progress_bar=prog)

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
    def run_engine_with_checkpointing(engine,lisaModel,rng_key,chain,checkpoint_file,Ntotal,checkpoint_at,resume=False):
        '''
        Runs the numpyro sampler with sampler state checkpointing. 
        
        Arguments
        -------------------
        [fill in]
        checkpoint_sampling (bool)  : When to checkpoint. Options:
                                        'end' (only saves sampler state at the very end of the run)
                                        'warmup' (saves after warmup phase and at end)
                                        'interval' (saves after warmup, at end, and after every checkpoint_interval number of samples)
                                        Note: Generally not worth checkpointing while sampling for large models/datasets, 
                                        as the recompliation and GPU off/onloading time will exceed the sampling time.

        resume (bool)               : Whether the run is being resumed. If so, skip the warmup phase.
        Returns
        --------------------
        post_samples (array)        : Posterior samples.
        
        '''
        

        if chain is None and not resume:
            if checkpoint_at=='warmup' or checkpoint_at=='interval':
                print("Beginning sampling, starting warmup phase...")
                ## run warmup phase
                engine.warmup(rng_key,lisaModel)
                state = engine.post_warmup_state
                print("Warmup phase complete. Checkpointing before initializing sampling...")
                ## save
                if dill.pickles([engine,state,chain]):
                    temp_file = checkpoint_file + ".temp"
                    with open(temp_file, "wb") as file:
                        pickle.dump([engine,state,chain], file)
                    shutil.move(temp_file, checkpoint_file)
                else:
                    print("WARNING: Cannot write checkpoint file, job cannot resume if interrupted.")
                
                engine.post_warmup_state = state
                rng_key = engine.post_warmup_state.rng_key
            ## warn if the checkpointing spec is wonky, but continue on as if it were 'end'
            elif checkpoint_at!='end':
                print("WARNING: Invalid specification of checkpointing behavior (checkpoint_at='{}'). Sampler state will be saved at end of sampling.".format(checkpoint_at))
        
        print("Initializing sampling...")
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
            
            if checkpoint_at=='interval':
                ## check to see if we have the desired number of samples yet
                Ncurrent = len(chain['theta_transformed'][0])
                if Ncurrent >= Ntotal:
                    break
            else:
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
        
        ## save the final state
        print("Sampling complete. Saving final sampler state to {}".format(checkpoint_file))
        if dill.pickles([engine,state,chain]):
            temp_file = checkpoint_file + ".temp"
            with open(temp_file, "wb") as file:
                pickle.dump([engine,state,chain], file)
            shutil.move(temp_file, checkpoint_file)
        else:
            print("Warning: Failed to save final state to checkpoint file, cannot resume sampling later.")
    
        ## retrive samples and reformat
        post_samples = np.array(chain['theta_transformed']).T
        
        return post_samples


#class numpyro_nested_engine():
#
#    '''
#    Class for interfacing with the Numpyro nested sampling sampler. 
#    '''
#    
#    @classmethod
#    def define_engine(cls, lisaobj, Nsamples, seed, gpu=False):
#        
#        if seed is not None:
#            rng_key = jax.random.PRNGKey(seed)
#        else:
#            raise TypeError("Numpyro sampler requires a defined seed.")
#        
#        constructor_kwargs = {'verbose':True,'parameter_estimation':True, 'num_live_points':800, 'max_samples':10*Nsamples}
#        termination_kwargs = {'dlogZ':1e-4}
#        ## if there are phase parameters, use the sph wrapper
#        if len(lisaobj.Model.blm_phase_idx) > 0:
#            engine = NestedSampler(numpyro_model_sph, constructor_kwargs=constructor_kwargs, termination_kwargs=termination_kwargs)
#        ## otherwise use the standard one
#        else:
#            engine = NestedSampler(numpyro_model, constructor_kwargs=constructor_kwargs, termination_kwargs=termination_kwargs)
#        engine.num_samples = Nsamples
##        engine = MCMC(kernel,num_warmup=Nburn,num_samples=Nsamples,num_chains=Nthreads,chain_method=chain_method,progress_bar=prog)
#
#        # print npar
#        print("Npar = " + str(lisaobj.Model.Npar))
#
#        return engine, lisaobj.Model.parameters, rng_key
#    
#    @staticmethod
#    def run_engine(engine,lisaModel,rng_key,checkpoint_file):
#        
#        # -------------------- Run HMC sampler ---------------------------
#        print("Beginning sampling...")
#        engine.run(rng_key,lisaModel)
#        print("Sampling complete. Retrieving posterior and plotting results...")
#        ## retrive samples, resample from weighted samples, and reformat
#        _, new_key = jax.random.split(rng_key,2)
#        post_samples = np.array(engine.get_samples(new_key,num_samples=engine.num_samples)['theta_transformed']).T
#        
#        print("Sampling complete. Saving final sampler state to {}".format(checkpoint_file))
#        if dill.pickles(engine):
#            temp_file = checkpoint_file + ".temp"
#            with open(temp_file, "wb") as file:
#                pickle.dump(engine, file)
#            shutil.move(temp_file, checkpoint_file)
#        else:
#            print("Warning: Failed to save final state to checkpoint file, cannot resume sampling later.")
#    
#        
#        return post_samples
#    
#    
#    def load_engine(resume_file):
#        
#        ## load model and parameters from previous checkpoint
#        if os.path.isfile(resume_file):
#            print("Loading interrupted analysis from last checkpoint...")
#            with open(resume_file,'rb') as file:
#                engine = pickle.load(file)
#        else:
#            raise TypeError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
#    
#        return engine
#    