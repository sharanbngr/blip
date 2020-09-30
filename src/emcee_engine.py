import numpy as np
import emcee


class emcee_engine():

    '''
    Class for interfacing with dynesty sampler. This method also contains the
    priors definition for all models written to work with the dynesty sampler.
    '''

    @classmethod
    def logpost(cls, theta, logprior, loglike):

        ''' Calculate log-posterior at theta''' 

        return logprior(theta) + loglike(theta)


    @classmethod
    def define_engine(cls, lisaobj, params, nlive, randst):

        # number of ensemble points
        #nlive = 100   

        # create the nested sampler objects
        if params['modeltype']=='isgwb':

            print("Doing an isotropic stochastic analysis...")
            parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$', r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            npar = len(parameters)

            logprior, loglike = cls.isgwb_prior, lisaobj.isgwb_log_likelihood

            # initialize walker positions
            init_samples = [ np.random.uniform(-43, -39, nlive), np.random.uniform(-50, -46, nlive), \
                 np.random.uniform(-4, 4, nlive),  np.random.uniform(-11, -4, nlive) ]
            init_samples = np.array(init_samples).T


        elif params['modeltype']=='sph_sgwb':

            print("Doing a spherical harmonic stochastic analysis ...")

            # add the basic parameters first
            parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$', r'$\alpha$', r'$\log_{10} (\Omega_0)$']

            # add the blms
            for lval in range(1, params['lmax'] + 1):
                for mval in range(lval + 1):

                    if mval == 0:
                        parameters.append(r'$b_{' + str(lval) + str(mval) + '}$' )
                    else:
                        parameters.append(r'$|b_{' + str(lval) + str(mval) + '}|$' )
                        parameters.append(r'$\phi_{' + str(lval) + str(mval) + '}$' )

            ## RM is line later.
            # parameters.append(r'$|b_{' + str(1) + str(1) + '}|$' )
            # parameters.append(r'$\phi_{' + str(1) + str(1) + '}$' )
            npar = len(parameters)

            logprior, loglike = cls.sph_prior, lisaobj.sph_log_likelihood


        elif params['modeltype']=='noise_only':

            print("Doing an instrumental noise only analysis ...")
            parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
            npar = len(parameters)

            # initialize walker positions
            init_samples = [ np.random.uniform(-43, -39, nlive), np.random.uniform(-50, -46, nlive) ]
            init_samples = np.array(init_samples).T

            logprior, loglike = cls.instr_prior, lisaobj.instr_log_likelihood

        elif params['modeltype'] =='isgwb_only':

            print("Doing an isgwb signal only analysis ...")
            parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            npar = len(parameters)

            logprior, loglike = cls.isgwb_only_prior, lisaobj.isgwb_only_log_likelihood

            # initialize walker positions
            init_samples = [ np.random.uniform(-4, 4, nlive),  np.random.uniform(-11, -4, nlive) ]
            init_samples = np.array(init_samples).T


        else:
            raise ValueError('Unknown recovery model selected')

        # print npar
        print("npar = " + str(npar))

        # the prior and the likelihood functions are arguments to the posterior
        argslist = (logprior, loglike)

        # set up the sampler
        engine = emcee.EnsembleSampler(nlive, npar, cls.logpost, args=argslist)


        return engine, parameters, init_samples

    @staticmethod
    def run_engine(engine, init_samples):

        Nburnin = 1000   # number of burn-in samples
        Nsamples = 200  # number of final posterior samples

        # pass the initial samples and total number of samples required
        engine.run_mcmc(init_samples, Nsamples + Nburnin, progress=True)
        #engine.sample(init_samples, iterations= Nsamples + Nburnin, store=True)

    
        ndims = engine.chain.shape[-1]

        # extract the samples (removing the burn-in)
        post_samples = engine.chain[:, Nburnin:, :].reshape((-1, ndims))

        return post_samples


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

        # unpack the model parameters from the tuple
        log_Np, log_Na = theta

        if (-43 < log_Np < -39) and (-50 < log_Na < 46):
            lp  = - (log_Np + log_Na) * np.log(10)

        else:
            lp =  -np.inf
        
        return lp


    @staticmethod
    def isgwb_prior(theta):


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


        # Unpack: Theta 
        log_Np, log_Na, alpha, log_omega0 = theta

        if (-43 < log_Np < -39) and (-50 < log_Na < -46) and (-11 < log_omega0 < -4) and (-4 < alpha < 4)  :
            lp  = - (log_Np + log_Na + log_omega0 ) * np.log(10)

        else:
            lp =  -np.inf
        
        return lp


    @staticmethod
    def sph_prior(self, theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref for each of the harmonics, Np and Na. The first element is always alpha and the last two are always Np and Na
        '''


        # Unpack: Theta 
        log_Np, log_Na, alpha, log_omega0 = theta

        if (-43 < log_Np < -39) and (-50 < log_Na < -46) and (-11 < log_omega0 < -4) and (-4 < alpha < 4)  :
            lp  = - (log_Np + log_Na + log_omega0 ) * np.log(10)

        else:
            return  -np.inf


        # The rest of the priors define the blm parameter space
        blm_theta = []

        ## counter for the rest of theta
        cnt = 4

        for lval in range(1, self.params['lmax'] + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    blm_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:
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

        # Unpack: Theta 
        alpha, log_omega0 = theta

        if (-11 < log_omega0 < -4) and (-4 < alpha < 4)  :
            lp  = - ( log_omega0 ) * np.log(10)

        else:
            lp =  -np.inf
        
        return lp
