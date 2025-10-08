import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
import random

def loadRunDir(rundir):
    with open(rundir +'config.pickle','rb') as paramfile:
        params = pickle.load(paramfile)
        inj = pickle.load(paramfile)
        parameters = pickle.load(paramfile)
    post = np.loadtxt(rundir +"/post_samples.txt")
    return params, post, parameters, inj

def quickMapmaker(params, sample, parameters, inj, nside=32, saveto=None):
    
    if type(parameters) is dict:
        blm_start = len(parameters['noise']) + len(parameters['signal'])
        ## deal with extra parameter in broken_powerlaw:
        if 'spectrum_model' in params.keys():
            if params['spectrum_model']=='broken_powerlaw':
                blm_start = blm_start - 1
        
    elif type(parameters) is list:
        print("Warning: using a depreciated parameter format. Number of non-b_lm parameters is unknown, defaulting to n=4.")
        blm_start = 4
    else:
        raise TypeError("parameters argument is not dict or list.")

    # size of the blm array
    blm_size = Alm.getsize(params['lmax'])

    blmax = params['lmax']  

    ## blms.
    blms_sample = np.append([1], sample[blm_start:])

    # Omega at 1 mHz
    # handle various spectral models, but default to power law
    ## include backwards compatability check (to be depreciated later)
    if 'spectrum_model' in params.keys():
        if params['spectrum_model']=='broken_powerlaw':
            alpha_1 = sample[2]
            log_A1 = sample[3]
            alpha_2 = sample[2] - 0.667
            log_A2 = sample[4]
            Omega_1mHz= ((10**log_A1)*(1e-3/params['fref'])**alpha_1)/(1 + (10**log_A2)*(1e-3/params['fref'])**alpha_2)
        else:
            if params['spectrum_model'] != 'powerlaw':
                print("Unknown spectral model. Defaulting to power law...")
            alpha = sample[2]
            log_Omega0 = sample[3]
            Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)
    else:
        print("Warning: running on older output without specification of spectral model.")
        print("Warning: defaulting to power law spectral model. This may result in unintended behavior.")
        alpha = sample[2]
        log_Omega0 = sample[3]
        Omega_1mHz = (10**(log_Omega0)) * (1e-3/params['fref'])**(alpha)

    ## Complex array of blm values for both +ve m values
    blm_vals = np.zeros(blm_size, dtype='complex')

    ## this is b00, alsways set to 1
    blm_vals[0] = 1
    cnt = 1

    for lval in range(1, blmax + 1):
        for mval in range(lval + 1):

            idx = Alm.getidx(blmax, lval, mval)

            if mval == 0:
                blm_vals[idx] = blms_sample[cnt]
                cnt = cnt + 1
            else:
                ## prior on amplitude, phase
                blm_vals[idx] = blms_sample[cnt] * np.exp(1j * blms_sample[cnt+1])
                cnt = cnt + 2

    norm = np.sum(blm_vals[0:(blmax + 1)]**2) + np.sum(2*np.abs(blm_vals[(blmax + 1):])**2)

    Omega_map  =  Omega_1mHz * (1.0/norm) * (hp.alm2map(blm_vals, nside))**2

    return Omega_map

def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)
    return list_object

def FWxM(skymap,fracMax=.5,nside=32,manual_peak_index=None,global_peak_val=None,ommission=[]):
    ommittedSkymap = delete_multiple_element(list(skymap),list(ommission))
    if not manual_peak_index:
        peak_value = max(ommittedSkymap)
        peak_index = list(skymap).index(peak_value)
        n=1
        while peak_index in ommittedSkymap:
            indices = [i for i, e in enumerate(skymap) if e == peak_value]
            peak_index = indices[n]
            n+=1
    else:
        peak_value=list(skymap)[int(manual_peak_index)]
        peak_index = manual_peak_index
    if not global_peak_val:
        global_peak_val = max(skymap)
    
    signalBlob = {peak_index}
    border = set(hp.pixelfunc.get_all_neighbours(nside,peak_index))
    count = 0
    full = False
    while not full:
        full = True
        for pxl_index in border:
            if skymap[pxl_index] > fracMax*global_peak_val:
                signalBlob = signalBlob.union({pxl_index})
                border = border.union(set(hp.pixelfunc.get_all_neighbours(nside,pxl_index))).difference(signalBlob)
                count+=1
                full = False       
    return count/len(skymap), signalBlob, peak_value, border

#Currently designed for one-point analysis
def getAngRes(params, sample, parameters, inj, makePlot=False,nside=32,fracMax=.5):
    Omega_map = quickMapmaker(params, sample, parameters, inj, nside)
    pointSize, signalBlob, peak_val, border = FWxM(Omega_map, fracMax=fracMax, nside=nside)

    if inj['injtype']=='point_source':
        condition_1 = True # checks to see if there are any pixels outside of the FWHM that are > half the peak value
        for pxl_val in delete_multiple_element(list(Omega_map),list(signalBlob)):
            if pxl_val > fracMax*peak_val: #you might want to change this line. maybe 10% or 20%?
                condition_1 = False
                break
        
        condition_2 = False # checks that the injected location is inside the FWHM -- you'll probably want to remove this part
        if hp.pixelfunc.ang2pix(nside,inj['theta'], inj['phi']) in signalBlob:
            condition_2 = True

        if condition_1 and condition_2:
            return pointSize, True
        else:
            return pointSize, False
    else:
        # print('Not a point source injection! n-point source injection analyses coming soon...')
        return pointSize,False

def FWxM_contour(omega_map,outdir=None,centers=[],fracMax=.5,coord='E',filename='localization_skymap',plotTitle='Localization Skymap',nside=32,maxBlobNumber=10,global_peak_val=None):
    giggaBlob = set()
    giggaBorder = set()
    hp.mollview(omega_map, coord=coord, title=plotTitle)
    if global_peak_val is None:
        global_peak_val=max(omega_map)
    if centers == []:
        peaks = []
        blobNumber = 0
        while True and blobNumber <= maxBlobNumber:
            pointSize, signalBlob, peak_val, border = FWxM(omega_map, fracMax=fracMax, nside=nside,global_peak_val=global_peak_val,ommission=giggaBlob)
            giggaBlob = giggaBlob.union(signalBlob)
            if peak_val < fracMax*global_peak_val:
                break
            giggaBorder = giggaBorder.union(border)
            peaks.append(peak_val)
        for peak in peaks:
            th0, ph0 = hp.pixelfunc.pix2ang(nside, list(omega_map).index(peak))
            hp.projscatter(th0, ph0, color='r', marker='*', coord=coord)
    else:
        for center in centers:
            pointSize, signalBlob, peak_val, border = FWxM(omega_map, fracMax, nside,manual_peak_index=center,global_peak_val=global_peak_val)
            giggaBlob = giggaBlob.union(signalBlob)
            giggaBorder = giggaBorder.union(border)
            th0, ph0 = hp.pixelfunc.pix2ang(nside, center)
            hp.projscatter(th0, ph0, color='r', marker='*', coord=coord)
    for pxl_index in giggaBorder:
        th, ph = hp.pixelfunc.pix2ang(nside, pxl_index)
        hp.projscatter(th, ph, color='r', marker='.', coord=coord)

    if outdir:
        plt.savefig(outdir +  filename +'.png', dpi=150)
    else:
        plt.savefig(filename +'.png', dpi=150)

def getLocalizationSummary(rundir,outdir=None,summary_filename='localization_summary',FWxM_filename='FWxM_list'):
    if outdir is None:
        outdir = rundir
    params, post, parameters, inj = loadRunDir(rundir)

    medianPointSize = getAngRes(params, np.median(post, axis=0), parameters, inj)
    meanPointSize = getAngRes(params, np.average(post, axis=0), parameters, inj)

    random.shuffle(post)
    areas=[]
    good=0
    count=0
    r=0
    print("There are " ,len(post)," samples for this run")
    for sample in post:
        area,quality = getAngRes(params, sample, parameters, inj)
        areas.append(area)
        if quality:
            good+=1
        if count <= r*len(post) < count+1:
            print(str(int(r*100+.1)) + '%')
            r+=.01
        count+=1
    recovery_quality = good/len(post)

    print('100%')

    confidence68 = [np.quantile(areas, .16),np.quantile(areas, .84)]
    confidence90 = [np.quantile(areas, .05),np.quantile(areas, .95)] 
    confidence95 = [np.quantile(areas, .025),np.quantile(areas, .975)]

    with open(outdir + summary_filename + '.txt','w') as f:
        f.write("Recovery quality (fraction of maps that satisfy conditions):" + str(recovery_quality) +  '\n')
        f.write("distribution mean FWxM: " + str(np.mean(areas)) +  '\n')
        f.write("95'%' confidence interval FWxM: " + str(confidence95) + '\n')
        f.write('\n')
        f.write("b_lm median FWxM: " + str(medianPointSize) +  '\n')
        f.write("b_lm mean FWxM: " + str(meanPointSize) +  '\n')
        f.write("distribution median FWxM: " + str(np.median(areas)) +  '\n')
        f.write("68'%' confidence interval FWxM: " + str(confidence68) +  '\n')
        f.write("90'%' confidence interval FWxM: " + str(confidence90) + '\n')

        
    with open(outdir + FWxM_filename + '.txt','w') as f:
        f.write(str(areas))
    
    FWxM_contour(quickMapmaker(params, np.median(post, axis=0), parameters, inj, 32),plotTitle='Median blm Localization Skymap',outdir=outdir,filename='mediuan_blm_localization_skymap')


if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='getLocalizationSummary', usage='%(prog)s [options] rundir', description='run getLocalizationSummary')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    parser.add_argument('-o','--outdir', default=None, type=str, help='The path to the out directory')

    # execute parser
    args = parser.parse_args()

    
    getLocalizationSummary(args.rundir,outdir=args.outfir)