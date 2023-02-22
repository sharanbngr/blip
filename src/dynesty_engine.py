import numpy as np
from dynesty import NestedSampler
from dynesty.utils import resample_equal
import dill
import time
import shutil, os
#
#from hierarchical import hierarchy

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
                signal_parameters = [r'$\log_{10} (\Omega_0)$',r'$\alpha_1$',r'$\alpha_2$',r'$\log_{10} (f_{break})$']
            elif params['spectrum_model']=='truncated_broken_powerlaw':
                signal_parameters = [r'$\log_{10} (\Omega_0)$',r'$\alpha_1$',r'$\alpha_2$',r'$\log_{10} (f_{\mathrm{break}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            elif params['spectrum_model']=='truncated_powerlaw':
                signal_parameters = [r'$\log_{10} (\Omega_0)$',r'$\alpha$', r'$\log_{10} (f_{\mathrm{break}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            elif params['spectrum_model']=='free_broken_powerlaw':
                signal_parameters = [r'$\log_{10} (A_1)$',r'$\alpha_1$',r'$\log_{10} (A_2)$',r'$\alpha_2$']
            else:
                raise ValueError("Unknown specification of spectral model. Available options: powerlaw, broken_powerlaw, and free_broken_powerlaw.")
        else:
            signal_parameters = []
        
        # create the nested sampler objects      
        if params['modeltype']=='isgwb':

            print("Doing an isotropic stochastic analysis...")
            all_parameters = noise_parameters + signal_parameters
            parameters = {'noise':noise_parameters,'signal':signal_parameters,'blm':[],'all':all_parameters}
            npar = len(all_parameters)
            if params['spectrum_model']=='powerlaw':
                engine = NestedSampler(lisaobj.isgwb_pl_log_likelihood, cls.isgwb_pl_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='broken_powerlaw':
                engine = NestedSampler(lisaobj.isgwb_bpl_log_likelihood, cls.isgwb_bpl_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='broken_powerlaw_2':
                engine = NestedSampler(lisaobj.isgwb_bpl2_log_likelihood, cls.isgwb_bpl2_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='truncated_broken_powerlaw':
                engine = NestedSampler(lisaobj.isgwb_tbpl_log_likelihood, cls.isgwb_tbpl_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='truncated_powerlaw':
                engine = NestedSampler(lisaobj.isgwb_tpl_log_likelihood, cls.isgwb_tpl_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
            elif params['spectrum_model']=='free_broken_powerlaw':
                engine = NestedSampler(lisaobj.isgwb_fbpl_log_likelihood, cls.isgwb_fbpl_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
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
            
            if 'hierarchy' in params.keys() and params['hierarchy']=='fg_scale_heights_full':
                hyperparameters = [r'$r_{h}$',r'$z_{h}$']
                blm_parameters = []
            else:
                hyperparameters = []
            all_parameters = noise_parameters + signal_parameters + hyperparameters + blm_parameters
            parameters = {'noise':noise_parameters,'signal':signal_parameters,'hyper':hyperparameters,'blm':blm_parameters,'all':all_parameters}
            npar = len(all_parameters)
            
            ## backward compatibility
            if 'hierarchy' in params.keys() and params['hierarchy']=='fg_scale_heights_full':
                if params['spectrum_model']=='truncated_powerlaw':
                    ## overwrite parameter count since we won't be directly sampling the blms
                    npar = len(noise_parameters + signal_parameters + hyperparameters)
#                    cls.hierarchical_engine = hierarchy(params)
                    engine = NestedSampler(lisaobj.sph_h_tpl_log_likelihood, cls.sph_h_tpl_prior, \
                                           npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
                else:
                    raise ValueError("The (prototype) hierarchical foreground analysis only currently supports a truncated power law spectral distribution for the at-runtime hierarchical analysis. Sorry!")
            else:
                if 'hierarchy' in params.keys() and params['hierarchy']=='fg_scale_heights_post':
                    print("Hierarchical foreground analysis will be performed in post-processing. Performing standard spherical harmonic search.")
            
                if params['spectrum_model']=='powerlaw':
                    engine = NestedSampler(lisaobj.sph_pl_log_likelihood, cls.sph_pl_prior,\
                        npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
                elif params['spectrum_model']=='broken_powerlaw':
                    engine = NestedSampler(lisaobj.sph_bpl_log_likelihood, cls.sph_bpl_prior,\
                        npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
                elif params['spectrum_model']=='broken_powerlaw_2':
                    engine = NestedSampler(lisaobj.sph_bpl2_log_likelihood, cls.sph_bpl2_prior,\
                        npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
                elif params['spectrum_model']=='truncated_broken_powerlaw':
                    engine = NestedSampler(lisaobj.sph_tbpl_log_likelihood, cls.sph_tbpl_prior,\
                        npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
                elif params['spectrum_model']=='truncated_powerlaw':
                    engine = NestedSampler(lisaobj.sph_tpl_log_likelihood, cls.sph_tpl_prior,\
                        npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
                elif params['spectrum_model']=='free_broken_powerlaw':
                    engine = NestedSampler(lisaobj.sph_fbpl_log_likelihood, cls.sph_fbpl_prior,\
                        npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size, rstate = randst)
                else:
                    raise ValueError("Unknown specification of spectral model. Available options: powerlaw, broken_powerlaw, free_broken_powerlaw, broken_powerlaw_2, truncated_broken_powerlaw, and truncated_powerlaw.")

        elif params['modeltype']=='noise_only':

            print("Doing an instrumental noise only analysis ...")
            noise_parameters = [r'$\log_{10} (Np)$', r'$\log_{10} (Na)$']
            parameters = {'noise':noise_parameters,'signal':[],'blm':[],'all':noise_parameters}
            npar = len(noise_parameters)

            engine = NestedSampler(lisaobj.instr_log_likelihood,  cls.instr_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size,  rstate = randst)

        elif params['modeltype'] =='isgwb_only':

            print("Doing an isgwb signal only analysis ...")
            signal_parameters = [r'$\alpha$', r'$\log_{10} (\Omega_0)$']
            parameters = {'noise':[],'signal':signal_parameters,'blm':[],'all':signal_parameters}
            npar = len(signal_parameters)

            engine = NestedSampler(lisaobj.isgwb_only_log_likelihood, cls.isgwb_only_prior,\
                    npar, bound='multi', sample='rwalk', nlive=nlive, pool=pool, queue_size=pool_size,  rstate = randst)

        else:
            raise ValueError('Unknown recovery model selected')

        # print npar
        print("npar = " + str(npar))

        return engine, parameters
    
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
        log_fbreak = -1*theta[5] - 2.5
        

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
#        log_fcut = theta[6] - 3.3
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
        lmax = np.sqrt( theta[4:].size + 1 ) - 1

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
        lmax = np.sqrt( theta[5:].size + 1 ) - 1

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
        lmax = np.sqrt( theta[6:].size + 1 ) - 1

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
        log_fcut = theta[6] - 3.3
        log_fscale = -1.7*theta[7] - 2.3
        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[8:]) + 1 ) - 1

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

        theta = [log_Np, log_Na, log_omega0, alpha_1, alpha_2, log_fbreak, log_fcut, log_fscale] + blm_theta

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
    
    @classmethod
    def sph_h_tpl_prior(cls,theta):

        '''
        Prior for a power spectra based spherical harmonic anisotropic analysis with a truncated power law spectral model,
            incorporating a hierarchical hyperprior on the spatial distribution from the Breivik+2020 parameterized Milky Way model. 

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
        
        ## hyperparameters
        rh = 2*theta[6] + 2
        zh = 1.95*theta[7] + 0.05
        
        theta = [log_Np, log_Na, log_omega0, alpha, log_fbreak, log_fscale, rh, zh]

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




