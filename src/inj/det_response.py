from __future__ import division
import numpy as np
import scipy.signal as sg
from src.makeGaussianData import gaussianData
import pdb

def Antennapatterns():

    '''
    Class  of the LISA array or of parts of it. The methods here include calculation of antenna patters for a single doppler channel, for
    the three michelson channels or for the AET TDI channels
    '''

    def doppler_response(self, f0, theta, phi):
        
        '''
        Calculate antenna patterns given theta and phi for the doppler channel of a stationary LISA. 
        Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and should be
        passed as an array
        '''


        ct = np.cos(theta)
        
        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)

        # Initlize arrays for the detector reponse
        Rplus, Rcross = np.zeros(f0.size), np.zeros(f0.size)

        # Calculate the detector response for each frequency

        # Calculate GW transfer function for the michelson channels
        gammaU    =    1/2 * (np.sinc(f0*(1 - udir))*np.exp(-1j*f0*(3+udir)) + \
                            np.sinc(f0*(1 + udir))*np.exp(-1j*f0*(1+udir)))


        ## Michelson Channel Antenna patterns for + pol: Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

        Rplus   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                            0.5*((np.cos(phi))**2 - ct**2))*gammaU

        ## Michelson Channel Antenna patterns for x pol
        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross

        Rcross = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU

        return Rplus, Rcorss


    def michelson_response(self, f0, theta, phi): 

        '''
        Calculate antenna patterns given theta and phi for the three michelson of a stationary LISA. 
        Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and should be
        passed as an array
        '''

        ct = np.cos(theta)

        ## udir is just u.r, where r is the directional vector
        udir = np.sqrt(1-ct**2) * np.sin(phi + np.pi/6)
        vdir = np.sqrt(1-ct**2) * np.sin(phi - np.pi/6)
        wdir = vdir - udir

        # Initlize arrays for the detector reponse
        R1plus, R1cross = np.zeros(f0.size)
        R2plus, R2cross = np.zeros(f0.size)
        R3plus, R3cross = np.zeros(f0.size)

        

        # Calculate GW transfer function for the michelson channels
        gammaU    =    1/2 * (np.sinc(f0*(1 - udir))*np.exp(-1j*f0*(3+udir)) + \
                         np.sinc((f0)*(1 + udir))*np.exp(-1j*f0*(1+udir)))

        gammaV    =    1/2 * (np.sinc(f0*(1 - vdir))*np.exp(-1j*f0*(3+vdir)) + \
                         np.sinc((f0)*(1 + vdir))*np.exp(-1j*f0*(1+vdir)))

        gammaW    =    1/2 * (np.sinc(f0*(1 - wdir))*np.exp(-1j*f0*(3+wdir)) + \
                         np.sinc((f0)*(1 + wdir))*np.exp(-1j*f0*(1+wdir)))

        ## Michelson Channel Antenna patterns for + pol
        ##  Fplus_u = 1/2(u x u)Gamma(udir, f):eplus

        Fplus_u   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 - np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2) + \
                        0.5*((np.cos(phi))**2 - ct**2))*gammaU

        Fplus_v   = 1/2*(1/4*(1-ct**2) + 1/2*(ct**2)*(np.cos(phi))**2 + np.sqrt(3/16)*np.sin(2*phi)*(1+ct**2)+ \
                     0.5*((np.cos(phi))**2 - ct**2))*gammaV

        Fplus_w   = 1/2*(1 - (1+ct**2)*(np.cos(phi))**2)*gammaW

        ## Michelson Channel Antenna patterns for x pol
        ##  Fcross_u = 1/2(u x u)Gamma(udir, f):ecross

        Fcross_u  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi + np.pi/3))*gammaU
        Fcross_v  = - np.sqrt(1-ct**2)/2 * (np.sin(2*phi - np.pi/3))*gammaV
        Fcross_w   = 1/2*ct*np.sin(2*phi)*gammaW

        ## Compelte Michelson antenna patterns
        ## Calculate Fplus
        R1plus = (Fplus_u - Fplus_v)
        R2plus = (Fplus_w - Fplus_u)
        R3plus = (Fplus_v - Fplus_w)

            ## Calculate Fcross
            R1cross = (Fcross_u - Fcross_v)
            R2cross = (Fcross_w - Fcross_u)
            R3cross = (Fcross_v - Fcross_w)

            return R1plus, R1corss, R2plus, R2cross, R3plus, R3corss

    def tdi_response(self, f0, theta, phi): 

        '''
        Calculate antenna patterns given theta and phi for the A,E and T channels of a stationary LISA. 
        Return the detector response for + and x polarization. Note that f0 is (pi*L*f)/c and should be
        passed as an array
        '''

        R1plus, R1corss, R2plus, R2cross, R3plus, R3corss  = self.michelson_response(f0, theta, phi)
        

        ## Calculate antenna patterns for the A, E and T channels
        RAplus = (2/3)*np.sin(2*f0)*(2*R1plus - R2plus - R3plus)
        REplus = (2/np.sqrt(3))*np.sin(2*f0)*(Fplus3 - R2plus)
        RTplus = (1/3)*np.sin(2*f0)*(R1plus + R3plus + R2plus)

        RAcross = (2/3)*np.sin(2*f0)*(2*R1cross - R2cross - R3cross)
        REcross = (2/np.sqrt(3))*np.sin(2*f0)*(R3cross - R2cross)
        RTcross = (1/3)*np.sin(2*f0)*(R1cross + R3cross + R2cross)

        return RAplus, RAcross, REplus, REcross, RTplus, RTcorss

    def tdi_isgwb_response(self, f0): 

        tt = np.arange(-1, 1, 0.01)
        pp = np.arange(0, 2*np.pi, np.pi/100)

        for ffs in f0:

            
            ## Detector response for the TDI Channels, summed over polarization
            ## and integrated over sky direction
            R1[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FAplus))**2 + (np.absolute(FAcross))**2)
            R2[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FEplus))**2 + (np.absolute(FEcross))**2)
            R3[ii] = dct*dphi/(4*np.pi)*np.sum((np.absolute(FTplus))**2 + (np.absolute(FTcross))**2)



        else:
            raise ValueError('Invalid Channel Name')

    return R1, R2, R3





def injectISGWB(Omega0=1e-10, alpha=0, fs=1, dur=1e5):

    '''
    Script to make time series SGWB injection for an equal arm LISA with no
    rotation or revolution around the sun. The output is the time series of
    the michelson channels. TDI is implemented by manipulating the time series
    later


    Only power law injections are possible for now of the information

    omega(f) = omega0 * (f/1mHz)^(alpha)

    Input:
    Omega0: The energy density at 1mHz
    alpha :  Power law index

    fs: sampling frequency of the desired noise

    dur: duration

    Output:
    Signal in michelson channels

    '''
    # speed of light
    cspeed = 3e8 #m/s

    # Length of the arms.
    L = 2.5e9

    freqs = np.logspace(-5, 0, 6000) # in Hz

    #Charactersitic frequency
    fstar = cspeed/(2*np.pi*L)

    # define f0 = f/2f*
    f0 = freqs/(2*fstar)



    ## There are the responses for the three arms
    ## Naturally RIJ = RJI

    R12, R23, R31 = calDetectorResponse(f0, 'Doppler')

    H0 = 2.2*10**(-18)
    Omegaf = Omega0*(freqs/(1e-3))**alpha

    # Spectrum of the SGWB
    Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2

    # Generate time series data for the doppler channels
    # First we have white gaussian data
    wht_data = np.random.normal(size=int(fs*dur))
    f_wht, wht_psd = sg.welch(wht_data, fs=fs, nperseg=int(fs*dur))


    # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
    # detector response tensor, and interpolated to the psd frequencies.
    S12_gw = np.interp(f_wht,freqs, Sgw*R12)
    S23_gw = np.interp(f_wht,freqs,Sgw*R23)
    S31_gw = np.interp(f_wht,freqs,Sgw*R31)

    S21_gw = np.interp(f_wht,freqs,Sgw*R12)
    S32_gw = np.interp(f_wht,freqs,Sgw*R23)
    S13_gw = np.interp(f_wht,freqs,Sgw*R31)

    # PSD of band-limited white gaussian noise
    N02 = 1.0/(f_wht[-1] - f_wht[0])

    ## Do an rrft
    wht_rfft = np.fft.rfft(wht_data)

    # Generate colored ffts
    h12_fft = wht_rfft*np.sqrt(S12_gw)/np.sqrt(N02)
    h23_fft = wht_rfft*np.sqrt(S23_gw)/np.sqrt(N02)
    h31_fft = wht_rfft*np.sqrt(S31_gw)/np.sqrt(N02)
    h21_fft = wht_rfft*np.sqrt(S21_gw)/np.sqrt(N02)
    h32_fft = wht_rfft*np.sqrt(S32_gw)/np.sqrt(N02)
    h13_fft = wht_rfft*np.sqrt(S13_gw)/np.sqrt(N02)

    h12_gw = np.fft.irfft(h12_fft, n=wht_data.size)
    h23_gw = np.fft.irfft(h23_fft, n=wht_data.size)
    h31_gw = np.fft.irfft(h31_fft, n=wht_data.size)
    h21_gw = np.fft.irfft(h21_fft, n=wht_data.size)
    h32_gw = np.fft.irfft(h32_fft, n=wht_data.size)
    h13_gw = np.fft.irfft(h13_fft, n=wht_data.size)



    if 0:
        import matplotlib.pyplot as plt
        from scipy.signal import welch
        import pdb

        delT =5e4
        nsegs = int(dur/delT)
        num = int(fs*delT)
        PSD12 = np.zeros((int(1+ num/2.0),nsegs))
        PSD23 = np.zeros((int(1+ num/2.0),nsegs))
        PSD31 = np.zeros((int(1+ num/2.0),nsegs))

        for ii in range(nsegs):
            idxmin = ii*num
            idxmax = idxmin + num

            fPSD, PSD12[:, ii] = welch(h12_gw[idxmin:idxmax], fs=fs, nperseg=num)
            fPSD, PSD23[:, ii] = welch(h23_gw[idxmin:idxmax], fs=fs, nperseg=num)
            fPSD, PSD31[:, ii] = welch(h31_gw[idxmin:idxmax], fs=fs, nperseg=num)


        fmin, fmax = 2e-4, 1e-1
        ymin, ymax = 1e-43, 0.8e-38
     
        plt.loglog(f_wht, S12_gw, label='required')
        plt.loglog(fPSD, np.mean(PSD12, axis=1),label='PSD12')
        plt.xlim(fmin, fmax)
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.savefig('PSD12.png', dpi=125)
        plt.close() 

        plt.loglog(f_wht, S23_gw, label='required')
        plt.loglog(fPSD, np.mean(PSD23, axis=1),label='PSD23')
        plt.xlim(fmin, fmax)
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.savefig('PSD23.png', dpi=125)
        plt.close() 

        plt.loglog(f_wht, S31_gw, label='required')
        plt.loglog(fPSD, np.mean(PSD31, axis=1),label='PSD31')
        plt.xlim(fmin, fmax)
        plt.ylim(ymin, ymax)
        plt.legend()
        plt.savefig('psd31.png', dpi=125)
        plt.close() 
        pdb.set_trace()


    # To generate michelson channels we need time shifts of multiples of L.
    tlag  = L/cspeed

    # time array and time shift array
    tarr = np.linspace(0, dur , num= h12_gw.size, endpoint=False)
    delt = tarr[2] - tarr[1]

    if 1.0/delt != fs:
        pdb.set_trace()

    # shifted time series
    tshift = tarr - tlag

    '''
    h1_gw = np.interp(tshift, tarr, h12_gw, left=h12_gw[0]) + h21_gw -\
    np.interp(tshift, tarr, h31_gw, left=h31_gw[0]) - h13_gw

    h2_gw = np.interp(tshift, tarr, h23_gw, left=h23_gw[0]) + h32_gw -\
    np.interp(tshift, tarr, h21_gw, left=h21_gw[0]) - h12_gw

    h3_gw = np.interp(tshift, tarr, h31_gw, left=h31_gw[0]) + h13_gw -\
    np.interp(tshift, tarr, h32_gw, left=h32_gw[0]) - h23_gw
    '''

    h1_gw, h2_gw, h3_gw = h12_gw - h13_gw, h23_gw - h21_gw,  h31_gw - h32_gw 

    if 0:
        diagM(h1_gw, h2_gw, h3_gw, fs, dur,Omega0, alpha)
    return h1_gw, h2_gw, h3_gw


### --------------------- Diagnostic function for TDI -----------------------------------------------------------------


def diagM(h1, h2, h3, fs, dur, Omega0, alpha):

    # Some code for testing if this ever breaks
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    import pdb

    freqs = np.arange(1e-4, 1e-1, 1e-4) # in Hz


    #Charactersitic frequency
    fstar = 3.0e8/(2*np.pi*2.5e9)

    # define f0 = f/2f*
    f0 = freqs/(2*fstar)

    R1, R2, R3 = calDetectorResponse(f0, 'Michelson')

    H0 = 2.2*10**(-18)
    Omegaf = Omega0*(freqs/(1e-3))**alpha

    # Spectrum of the SGWB
    Sgw = Omegaf*(3/(4*freqs**3))*(H0/np.pi)**2

    # Spectrum of the SGWB signal as seen in LISA data, ie convoluted with the
    # detector response tensor.
    S1_gw = Sgw*R1
    S2_gw = Sgw*R2
    S3_gw = Sgw*R3
        
    ### ---------------------------------- Check data PSD now --------------------------------------
    delT =5e4
    nsegs = int(dur/delT)
    num = int(fs*delT)
    PSD1 = np.zeros((int(1+ num/2.0),nsegs))
    PSD2 = np.zeros((int(1+ num/2.0),nsegs))
    PSD3 = np.zeros((int(1+ num/2.0),nsegs))

    for ii in range(nsegs):
        idxmin = ii*num
        idxmax = idxmin + num

        fPSD, PSD1[:, ii] = welch(h1[idxmin:idxmax], fs=fs, nperseg=num)
        fPSD, PSD2[:, ii] = welch(h2[idxmin:idxmax], fs=fs, nperseg=num)
        fPSD, PSD3[:, ii] = welch(h3[idxmin:idxmax], fs=fs, nperseg=num)

    fmin, fmax = 2e-4, 1e-1
    ymin, ymax = 1e-43, 0.8e-38

    plt.loglog(freqs, S1_gw, label='required')
    plt.loglog(fPSD, np.mean(PSD1, axis=1),label='PSD1')
    plt.xlim(fmin, fmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig('psd1.png', dpi=125)
    plt.close() 

    plt.loglog(freqs, S3_gw, label='required')
    plt.loglog(fPSD, np.mean(PSD3, axis=1),label='PSD3')
    plt.xlim(fmin, fmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig('psd3.png', dpi=125)
    plt.close() 

    plt.loglog(freqs, S2_gw, label='required')
    plt.loglog(fPSD, np.mean(PSD2, axis=1),label='PSD2')
    plt.xlim(fmin, fmax)
    plt.ylim(ymin, ymax)
    plt.legend()
    plt.savefig('psd2.png', dpi=125)
    plt.close() 

    pdb.set_trace()
