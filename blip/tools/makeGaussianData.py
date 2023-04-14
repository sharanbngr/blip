import numpy as np
#import matplotlib.pyplot as plt


def gaussianData(Sh,freqs, fs=1, dur=1e5):
    '''
    Script for generation time series data of a given spectral density.

    Input:
    Sh : desired spectral density
    freqs : corresponding frequencies
    fs : sampleRate in Hz
    dur : duration in seconds

    Output:
    Random time series data of duration dur with the prescribed spectrum

    Adapted from gaussian_noise.m from stamp
    '''
    # Number of data points in the time series
    N = int(fs*dur)

    # prepare for FFT
    if  np.mod(N,2)== 0 :
        numFreqs = N/2 - 1;
    else:
        numFreqs = (N-1)/2;

    # We will make an array of the desired frequencies
    delF = 1/dur
    fmin = 1/dur
    fmax = np.around(dur*fs/2)/dur
    delF = 1/dur

    # The output frequency series
    fout = np.linspace(fmin, fmax, numFreqs)

    # Interpolate to the desired frequencies
    norms = np.interp(fout, freqs, Sh)

    # Amplitude for for ifft
    norms = np.sqrt(norms*fs*N)/2.0

    # Normally distributed in frequency space
    re1 = norms*np.random.normal(size=fout.size)
    im1 = norms*np.random.normal(size=fout.size)

    htilda = re1 + 1j*im1

    if np.mod(N, 2) == 0:
        htilda = np.concatenate((np.zeros(1), htilda,np.zeros(1), np.flipud(np.conjugate(htilda))))
    else:
        htilda = np.concatenate((np.zeros(1),htilda, np.conjugate(np.flipud(htilda))))

    # Take inverse fft to get time series data
    ht = np.real(np.fft.ifft(htilda, N))

    doDiag=0
    if doDiag==1:

        # Some code for testing if this ever breaks
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.signal import welch
        from tools.printmap import printmap
        import pdb;

        delT =5e4
        nsegs = int(dur/delT)
        num = int(fs*delT)
        PSD = np.zeros((int(1+ num/2.0),nsegs))

        for ii in range(nsegs):
            idxmin = ii*num
            idxmax = idxmin + num

            fPSD, PSD[:, ii] = welch(ht[idxmin:idxmax], fs=fs, nperseg=num)

        PSD2 = np.mean(PSD, axis=1)
        plt.loglog(freqs, Sh, label='required')
        plt.loglog(fPSD, PSD2,label='mean from timeseries')
        #plt.loglog(fPSD, PSD[:, 1],label='from timeseries at 1')
        #plt.loglog(fPSD, PSD[:, 5],label='from timeseries at middle')
        plt.xlim(2e-4, 1e-1)
        plt.ylim(1e-42, 1e-38)
        plt.legend()
        plt.savefig('psds.png', dpi=125)
        plt.close()


        printmap(PSD/1e-42, [0, nsegs], [2e-4, 2e-1], 't', 'f', 'PSD Spectrogram', [0, 5], 'psd_spectrogram.png')        
        pdb.set_trace()

    return ht

