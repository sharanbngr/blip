import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

def printmap(map, xvals, yvals, xlab, ylab, clabel, crange, saveas):
    
    '''
    Auxially function for making a spectrogram map
    '''

    ax = plt.axes([0.15, 0.15, .75, .75])
    plt.imshow(map, cmap='hot', origin='lower', aspect='auto', extent=(xvals[0],xvals[1], yvals[0],yvals[1]))
    intervals = float(0.05)
    plt.xlabel(xlab, fontsize=13)
    plt.ylabel(ylab, fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(which='both',color='k', linestyle='--', linewidth=0.5, alpha=0.2)
    cb = plt.colorbar()
    plt.clim(crange[0], crange[1])
    cb.locator = ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    cb.set_label(clabel)
    cb.ax.tick_params(labelsize=13)
    plt.savefig(saveas, dpi=125)
    plt.close()
