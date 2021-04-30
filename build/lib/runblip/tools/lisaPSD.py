import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#mpl.rcParams['figure.figsize'] = (14,10)
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def lisaPSD(L=2.5e9, channel='TDI', fmin=5e-6, fmax=1e0, delf=1e-6, doPlot=False):
    '''
    This script calcualtes the power spectrum of LISA channels. It
    assumes an equal arm staionary LISA. If channel flag is 'M', then the
    michelson channels are given.


    If the channels flag is TDI, we implement TDI as
    described in http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308

    The noise levels are taken from the 2017 LISA proposal. The other input is
    the arm length of IFO, the default is 2.5 million km

    '''

    # speed of light
    cspeed = 3e8 #m/s

    frange = np.arange(fmin, fmax, delf) # in Hz

    numfreqs = frange.size

    #Charactersitic frequency
    fstar = cspeed/(2*np.pi*L)

    # define f0 = f/2f*
    f0 = frange/(2*fstar)


    Tyear = 365.24*24*60*60     # in seconds

    H0 = 2.2*10**(-18)         # in SI units, from the planck observations


    # 2017 Lisa proposal noise estimations

    # Position noise, converted to phase
    Sp = 4e-42*(1 + ((2e-3)/frange)**4)
    # Acceleration noise converted to phase
    Sa = 3.6e-49*(1+ ((4e-4)/(frange))**2)*(1+ (frange/(8e-3))**4)*(1/(2*np.pi*frange)**4)

    ###############################
    #   Old values from Adams and Cornish, 2010
    #
    #   Sp = 4e-42*(1+ 0*frange)
    #   Sa = 9e-50*(1+ ((1e-4)/(frange))**2)*(1/(2*np.pi*frange)**4)
    #
    ###################################


    ## Noise spectra of the TDI Channels
    if channel == 'TDI':
        # A channel
        S1  = (4/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*(12*Sp) + 24*Sp )+ \
        (16/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*12*Sa + np.cos(4*f0)*(6*Sa) + 18*Sa)
        # E  channel
        S2  = (4/3)*(np.sin(2*f0))**2*(4*Sp + (4 + 4*np.cos(2*f0))*Sp) +\
        (16/3)*(np.sin(2*f0))**2*(4*Sa + 4*Sa*np.cos(2*f0) +  4*Sa*(np.cos(2*f0))**2 )
        # T channel
        S3 = (4/9)*(np.sin(2*f0))**2* (12 - 12*np.cos(2*f0))*Sp + \
        (16/9)*(np.sin(2*f0))**2*(6 - 12*np.cos(2*f0) + 6*(np.cos(2*f0))**2)*Sa

        np.save('LISA_2017_PSD_TDI', (frange, S1,S2, S3 ))

        if doPlot:
            plt.plot(frange, S1, label = 'Channel A')
            plt.plot(frange, S2, label = 'Channel E')
            plt.plot(frange, S3, label = 'Channel T')
            plt.legend()
            plt.xlabel('Frequency')
            plt.ylabel('Noise spectrum sqrt(Hz) ')
            plt.gca().set_xlim(5*1e-4, 0.1)
            plt.gca().set_ylim(1e-49, 1e-38)
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('TDI_PSD.png', dpi=150)
            plt.close()




    elif channel == 'M': #Michelson Channels

        S1 = 4*Sp + 8*Sa*(1 + (np.cos(2*f0))**2)
        S2 = S1
        S3 = S1

        np.save('LISA_2017_PSD_M', (frange, S1,S2, S3 ))

        if doPlot:
            plt.plot(frange, S1, label = 'Michelson channel')
            plt.legend()
            plt.xlabel('Frequency')
            plt.ylabel('Noise spectrum sqrt(Hz) ')
            plt.gca().set_xlim(5*1e-4, 0.1)
            plt.gca().set_ylim(1e-43, 1e-37)
            plt.xscale('log')
            plt.yscale('log')
            plt.savefig('Mchannel_PSD.png', dpi=150)
            plt.close()
