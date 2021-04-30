import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb
import sys
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['figure.figsize'] = (12,6)
mpl.rcParams.update({'font.size': 18})


def stochasticSensitivity(L):
    '''
    This script calcualtes the sensititvity of LISA to an isotropic SGWB. It
    assumes an equal arm staionary LISA. Nevertheless we implement TDI as
    described in http://iopscience.iop.org/article/10.1088/0264-9381/18/17/308

    The noise levels are taken from the 2017 LISA proposal. The only input is
    the arm length of IFO, the default is 2.5 million km

    '''

    # speed of light
    cspeed = 3e8 #m/s

    #frange = np.logspace(-5, 0, 6000) # in Hz
    frange = np.arange(1e-5, 1e-1, 1e-5)
    
    # Calcualte the frequency bin width
    # dely  = np.log10(frange[1]/frange[0])
    # delf = frange*10**(dely)

    #Charactersitic frequency
    fstar = cspeed/(2*np.pi*L)

    # define f0 = f/2f*
    f0 = frange/(2*fstar)


    # cos(theta) space and phi space. The angular integral is a linear and
    # rectangular in the cos(theta) and phi space
    tt = np.arange(-1, 1, 0.01)
    pp = np.arange(0, 2*np.pi, np.pi/100)

    [ct, phi] = np.meshgrid(tt,pp)


    ## udir is just u.omega
    udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
    vdir = np.sqrt(1-ct**2) * np.sin(phi - np.pi/6)
    wdir = vdir - udir

    ## Intialize R,  the integrated transfer function
    # R1 = np.zeros(frange.size)
    # R2 = np.zeros(frange.size)
    # R3 = np.zeros(frange.size)

    RA = np.zeros(frange.size)
    RE = np.zeros(frange.size)
    RT = np.zeros(frange.size)

    for ii in range(0, f0.size):

        # Calculate GW transfer function for the michelson channels
        gammaU    =    1/2 * (np.sinc((f0[ii])*(1 - udir))*np.exp(-1j*f0[ii]*(3+udir)) + \
                         np.sinc((f0[ii])*(1 + udir))*np.exp(-1j*f0[ii]*(1+udir)))

        gammaV    =    1/2 * (np.sinc((f0[ii])*(1 - vdir))*np.exp(-1j*f0[ii]*(3+vdir)) + \
                         np.sinc((f0[ii])*(1 + vdir))*np.exp(-1j*f0[ii]*(1+vdir)))

        gammaW    =    1/2 * (np.sinc((f0[ii])*(1 - wdir))*np.exp(-1j*f0[ii]*(3+wdir)) + \
                         np.sinc((f0[ii])*(1 + wdir))*np.exp(-1j*f0[ii]*(1+wdir)))


        ## Michelson Channel Antenna patterns for + pol
        ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

        Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                        0.75*((np.cos(phi))**2 - ct**2))*gammaU

        Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)+ \
                     0.75*((np.cos(phi))**2 - ct**2))*gammaV

        Fplus_w   = 1/2*(1 - (1+ct**2)*(np.cos(phi))**2)*gammaW

        ## Calculate Fplus
        Fplus_1 = (Fplus_u - Fplus_v)
        Fplus_2 = (Fplus_w - Fplus_u)
        Fplus_3 = (Fplus_v - Fplus_w)


        ## Michelson Channel Antenna patterns for x pol
        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross

        Fcross_u  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU
        Fcross_v  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi - np.pi/3))*gammaV
        Fcross_w   = 1/2*ct*np.sin(2*phi)*gammaW

        ## Calculate Fcross
        Fcross_1 = (Fcross_u - Fcross_v)
        Fcross_2 = (Fcross_w - Fcross_u)
        Fcross_3 = (Fcross_v - Fcross_w)


        ## Calculate antenna patterns for the A, E and T channels
        FAplus = (2/3)*np.sin(2*f0[ii])*(2*Fplus_1 - Fplus_2 - Fplus_3)
        FEplus = (2/np.sqrt(3))*np.sin(2*f0[ii])*(Fplus_3 - Fplus_2)
        FTplus = (1/3)*np.sin(2*f0[ii])*(Fplus_1 + Fplus_3 + Fplus_2)

        FAcross = (2/3)*np.sin(2*f0[ii])*(2*Fcross_1 - Fcross_2 - Fcross_3)
        FEcross = (2/np.sqrt(3))*np.sin(2*f0[ii])*(Fcross_3 - Fcross_2)
        FTcross = (1/3)*np.sin(2*f0[ii])*(Fcross_1 + Fcross_3 + Fcross_2)


        dct = ct[0, 1] - ct[0,0]
        dphi = phi[1,0] - phi[0,0]

        ## Detector response for the Michelson Channels, summed over polarization
        ## and integrated over sky direction
        # R1[ii] = dct*dphi/(4*np.pi)*np.sum(np.sum((np.absolute(Fplus_1))**2 + (np.absolute(Fcross_1))**2))
        # R2[ii] = dct*dphi/(4*np.pi)*np.sum(np.sum((np.absolute(Fplus_2))**2 + (np.absolute(Fcross_2))**2))
        # R3[ii] = dct*dphi/(4*np.pi)*np.sum(np.sum((np.absolute(Fplus_3))**2 + (np.absolute(Fcross_3))**2))


        ## Detector response for the TDI Channels, summed over polarization
        ## and integrated over sky direction
        RA[ii] = dct*dphi/(4*np.pi)*np.sum(np.sum((np.absolute(FAplus))**2 + (np.absolute(FAcross))**2))
        RE[ii] = dct*dphi/(4*np.pi)*np.sum(np.sum((np.absolute(FEplus))**2 + (np.absolute(FEcross))**2))
        RT[ii] = dct*dphi/(4*np.pi)*np.sum(np.sum((np.absolute(FTplus))**2 + (np.absolute(FTcross))**2))


    Tyear = 365.24*24*60*60     # in seconds

    H0 = 2.2*10**(-18)         # in SI units, from the planck observations


    # 2017 Lisa proposal noise estimations

    # Position noise, converted to phase
    #Sp = 4e-42*(1 + ((2e-3)/frange)**4)
    # Acceleration noise converted to phase
    #Sa = 3.6e-49*(1+ ((4e-4)/(frange))**2)*(1+ (frange/(8e-3))**4)*(1/(2*np.pi*frange)**4)

    ###############################
    #   Old values from Adams and Cornish, 2010
    #
    Sp = 4e-42*(1+ 0*frange)
    Sa = 9e-50*(1+ ((1e-4)/(frange))**2)*(1/(2*np.pi*frange)**4)
    #
    ###################################

    ## Noise spectra of the TDI Channels

    SAA  = (4/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*(12*Sp) + 24*Sp )+ \
    (16/9)*(np.sin(2*f0))**2*(np.cos(2*f0)*12*Sa + np.cos(4*f0)*(6*Sa) + 18*Sa)

    SEE  = (4/3)*(np.sin(2*f0))**2*(4*Sp + (4 + 4*np.cos(2*f0))*Sp) +\
    (16/3)*(np.sin(2*f0))**2*(4*Sa + 4*Sa*np.cos(2*f0) +  4*Sa*(np.cos(2*f0))**2 )

    STT = (4/9)*(np.sin(2*f0))**2* (12 - 12*np.cos(2*f0))*Sp + \
    (16/9)*(np.sin(2*f0))**2*(6 - 12*np.cos(2*f0) + 6*(np.cos(2*f0))**2)*Sa


    SDplot = False



    if SDplot:
        plt.plot(frange, SAA, label = 'Channel A')
        plt.plot(frange, SEE, label = 'Channel E')
        plt.plot(frange, STT, label = 'Channel T')
        plt.legend()
        plt.xlabel('Frequency')
        plt.ylabel('Noise spectrum sqrt(Hz) ')
        plt.gca().set_xlim(5*1e-4, 0.1)
        plt.gca().set_ylim(1e-49, 1e-38)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('TDI_spectrum.png', dpi=150)
        plt.close()




    Omegaplot = False

    if Omegaplot:

        # Calculate Omega Sensitivity curves
        
        OmegaA = (4*np.pi*np.pi*(SAA/RA)*frange**3)/(3*np.sqrt(Tyear*delf)*H0**2)
        OmegaE = (4*np.pi*np.pi*(SEE/RA)*frange**3)/(3*np.sqrt(Tyear*delf)*H0**2)
        OmegaT = (4*np.pi*np.pi*(STT/RT)*frange**3)/(3*np.sqrt(Tyear*delf)*H0**2)
        
        
        plt.plot(frange, OmegaA, label='omega A')
        plt.plot(frange, OmegaE, label='omega E')
        plt.plot(frange, OmegaT, label='omega T')
        plt.legend()
        plt.xlabel('Frequency')
        plt.ylabel('Omega')
        plt.gca().set_xlim(10**(-5), 0.1)
        plt.gca().set_ylim(10**(-13), 10**(-6))
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('Omega_sensitivty.png', dpi=150)
        plt.close()




    sensplot = True

    

    if sensplot:
        omega_fid = 1e-10


        Sens_fid = (3*omega_fid*H0**2)/(4*np.pi*np.pi*frange**3)

        plt.plot(frange, np.sqrt(SAA/RA), label = 'Channel A, E', linewidth = 0.75)
        #plt.plot(frange, np.sqrt(SEE/RE), label = 'Channel E')
        #plt.plot(frange, np.sqrt(STT/RT), label = 'Channel T')
        plt.plot(frange, np.sqrt(Sens_fid), label=r'$\Omega_{gw} = 10^{-10}, \alpha = 0 $', linewidth=0.75)
        plt.legend()
        plt.xlabel('Frequency')
        plt.ylabel(r'$h \sqrt{Hz} $')
        plt.gca().set_xlim(5*1e-6, 0.2)
        plt.gca().set_ylim(1e-22, 1e-11)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('Sens_spectrum.png', dpi=150)

        plt.close()




        
#np.savetxt('LISA_2017_Omega.txt', (frange, OmegaA, OmegaE, OmegaT ))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError('Provide LISA arm length in meters as an argument')
    else:
        stochasticSensitivity(float(sys.argv[1]))
