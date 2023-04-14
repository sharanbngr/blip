import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import dill
import time
import shutil, os

class dynesty_engine():

    '''
    Class for interfacing with dynesty sampler. This method also contains the
    priors definition for all models written to work with the dynesty sampler.
    '''


    @classmethod
    def define_engine(cls, lisaobj, params, nlive, nthread, randst, pool=None, resume=False):

        # create multiprocessing pool
        if nthread > 1:
            pool_size = nthread
        else:
            if pool is not None:
                print("Warning: Nthread=1 but pool has been defined. This shouldn't happen...")
            pool = None
            pool_size = None
        
        
        
        engine = NestedSampler(lisaobj.Model.likelihood, lisaobj.Model.prior, lisaobj.Model.Npar,\
                    bound='multi', sample=params['sample_method'], nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
        
        
        # print npar
        print("Npar = " + str(lisaobj.Model.Npar))

        return engine, lisaobj.Model.parameters

    
    def load_engine(params,randst,pool):
        ## load engine from previous checkpoint
        ## nqueue is set to a negative number to trigger the queue to be refilled before the first iteration.
        ## randomstate cannot be saved, so we need to set that as well
        resume_file = params['out_dir']+'/checkpoint.pickle'
        if os.path.isfile(resume_file):
            print("Loading interrupted analysis from last checkpoint...")
            with open(resume_file,'rb') as file:
                engine, parameters = dill.load(file)
                if engine.added_live:
                    engine._remove_live_points()
                engine.nqueue = -1
                engine.rstate = randst
                if pool is not None:
                    engine.pool = pool
                    engine.loglikelihood.pool = pool
                    engine.M = engine.pool.map
                else:
                    engine.pool = None
                    engine.loglikelihood.pool = None
                    engine.M = map
        else:
            raise TypeError("Checkpoint file <{}> does not exist. Cannot resume from checkpoint.".format(resume_file))
        
        return engine, parameters
    
    @staticmethod
    def run_engine_with_checkpointing(engine,parameters,interval,checkpoint_file,step=1000):

       # -------------------- Run nested sampler ---------------------------
        pool = engine.pool
        old_ncall = engine.ncall
        start_time = time.time()
        while True:
            engine.run_nested(dlogz=0.5,maxiter=step,print_progress=True)
            if engine.ncall == old_ncall:
                break
            old_ncall = engine.ncall
            if os.path.isfile(checkpoint_file):
                last_checkpoint_s = time.time() - os.path.getmtime(checkpoint_file)
            else:
                last_checkpoint_s = time.time() - start_time
            if last_checkpoint_s > interval:
                print("Checkpointing...")
                ## pause the pool for saving
                if engine.pool is not None:
                    engine.pool = None
                    engine.M = map
                ## save
                if dill.pickles([engine,parameters]):
                    temp_file = checkpoint_file + ".temp"
                    with open(temp_file, "wb") as file:
                        dill.dump([engine,parameters], file)
                    shutil.move(temp_file, checkpoint_file)
                else:
                    print("Warning: Cannot write checkpoint file, job cannot resume if interrupted.")
                ## restart pool if needed
                if pool is not None:
                    engine.pool = pool
                    engine.M = engine.pool.map
                ## removes live points lumped in with dead points when previous sampling call concluded
                ## unsure why dynesty requires this, but it seems to be deeply important
                if engine.added_live:
                    engine._remove_live_points()
        # re-scale weights to have a maximum of one
        res = engine.results
        weights = np.exp(res['logwt'] - res['logz'][-1])
        weights[-1] = 1 - np.sum(weights[0:-1])

        post_samples = resample_equal(res.samples, weights)

        # Pull the evidence and the evidence error
        logz = res['logz']
        logzerr = res['logzerr']


        return post_samples, logz, logzerr
    
    @staticmethod
    def run_engine(engine):


       # -------------------- Run nested sampler ---------------------------
        engine.run_nested(dlogz=0.5,print_progress=True )
        
        # re-scale weights to have a maximum of one
        res = engine.results
        weights = np.exp(res['logwt'] - res['logz'][-1])
        weights[-1] = 1 - np.sum(weights[0:-1])

        post_samples = resample_equal(res.samples, weights)

        # Pull the evidence and the evidence error
        logz = res['logz']
        logzerr = res['logzerr']


        return post_samples, logz, logzerr




