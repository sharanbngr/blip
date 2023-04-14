import numpy as np
import emcee

class emcee_engine():

    '''
    Class for interfacing with dynesty sampler. This method also contains the
    priors definition for all models written to work with the dynesty sampler.
    '''

    @classmethod
    def logpost(cls, theta, prior_transform, loglike):

        '''
        Wrapper to calculate log-posterior at theta.
        
        Redefining this to place theta on the unit cube, consistent with the other implemented samplers.
        
        ''' 
        ## enforce theta on unit cube
        if np.any(theta > 1) or np.any(theta < 0):
            return -np.inf
        
        theta = prior_transform(theta)
        
        return loglike(theta)


    @classmethod
    def define_engine(cls, model, nlive, randst, pool=None):
        '''
        Defines the emcee engine.
        
        Arguments
        ----------
        model : Model object containing all relevant methods and parameters for evaluating the unified prior and likelihood
        nlive (int) : number of live points
        randst : initial random state
        
        '''

        # number of ensemble points
        #nlive = 100   
        
        parameters = model.parameters['all']
        
        Npar = model.Npar
        
        if 'seed' in model.params.keys():
            seed = model.params['seed']
        else:
            seed = None
        rng = np.random.default_rng(seed=seed)
        
        ## get initial samples on the unit cube
        init_samples = np.array([rng.uniform(0,1,nlive) for i in range(Npar)]).T
        
        # define moves
        moves = emcee.moves.StretchMove(a=2.0)
        #moves = emcee.moves.KDEMove(bw_method=.8)
        
        # set up the sampler
        engine = emcee.EnsembleSampler(nlive, Npar, cls.logpost, args=(model.prior, model.likelihood), moves=moves, pool=pool)


        return engine, parameters, init_samples


    @staticmethod
    def run_engine(engine, model, init_samples, Nburn, Nsamples):
        '''
        Runs the emcee sampler.
        
        Arguments
        ----------
        engine : the emcee EnsembleSampler engine
        init_samples Ndim x Nwalkers array) : walker start positions
        Nburn (int) : Number of burn-in samples
        Nsamples (int) : Number of final posterior samples.
        
        Returns
        ---------
        post_samples (array) : the final posterior samples
        
        '''

        # pass the initial samples and total number of samples required
        engine.run_mcmc(init_samples, Nsamples + Nburn, progress=True)
        #engine.sample(init_samples, iterations= Nsamples + Nburnin, store=True)

    
        ndims = engine.chain.shape[-1]

        # extract the samples (removing the burn-in)
        unit_samples = engine.chain[:, Nburn:, :].reshape((-1, ndims))
        
        ## apply prior transform to samples on the unit cube
        post_samples = np.vstack(model.prior([np.array(unit_samples[:,i]) for i in range(model.Npar)])).T

        return unit_samples, post_samples
