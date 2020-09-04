import init_oiii as init
import pymultinest
import multiprocessing as mp
import numpy as np
import sys
import os
import shutil
import glob
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Define a start time of the run.
start_time = time.time()

# Get a listing of the spectra involved in this run.
basepath = init.out_dir
specpath = init.spec_dir
target = init.target
listing = sorted(os.listdir(specpath))

# Provide some names and values for the lines that are being fit here.
lines = {
    'H-beta': 4861.3,
    '[O III] 4959': 4958.92,
    '[O III] 5007': 5006.84}


def make_dirs():
    """make directory structure for output data"""
    if not os.path.exists(os.path.join(basepath, f'{target}_out')):
        os.mkdir(os.path.join(basepath, f'{target}_out'))

    if not os.path.exists(os.path.join(basepath, f'{target}_out', 'out')):
        os.mkdir(os.path.join(basepath, f'{target}_out', 'out'))

    if not os.path.exists(os.path.join(basepath, f'{target}_out', 'plots')):
        os.mkdir(os.path.join(basepath, f'{target}_out', 'plots'))

#==============================================================================
# Define the Gaussian models.
#==============================================================================


def make_gaussian(mu, sigma, flux):
    h = flux / (sigma * np.sqrt(2*np.pi))
    s = -1.0 / (2 * sigma**2)

    def f(x):
        return h * np.exp(s * (x - mu)**2)
    return f


def o3gauss(pos1, width1, flux1):
    # [O III] is a doublet with a relationship between the two lines.
    # Define the second Gaussian in terms of the first.
    pos2 = (pos1 - ((1 + init.red)
                    * (lines['[O III] 5007'] - lines['[O III] 4959'])))
    width2 = width1
    flux2 = flux1 / 3
    # make and return the combined Gaussian
    gauss1 = make_gaussian(pos1, width1, flux1)
    gauss2 = make_gaussian(pos2, width2, flux2)
    return gauss1(x) + gauss2(x)

#==============================================================================
# Define prior, model, and loglikelihood functions that work for any number
# of components.
#==============================================================================


def prior(cube, ndim, nparams):
    if ((ndim != 1) and (ndim % 3 != 0)) or (ndim <= 0):
        raise ValueError(
            ('The number of dimensions must be positive and equal to '
             f'1 or a multiple of 3. ndim={ndim}'))
    if ndim == 1:
        cube[0] = avg
    else:
        for i in range(0, ndim, 3):
            # uniform wavelength prior
            cube[i+0] = (init.minwave + cube[i+0]
                         * init.wave_range)
             # uniform width prior
            cube[i+1] = (init.minwidth
                         + cube[i+1]*(init.maxwidth - init.minwidth))
            # log-uniform flux prior
            cube[i+2] = (threesigma * cube[i+1] * np.sqrt(2*np.pi)
                         * np.power(10, cube[i+2] * 4))


def model(*args):
    nargs = len(args)
    if ((nargs != 1) and (nargs % 3 != 0)) or (nargs <= 0):
        raise ValueError(
            ('The number of arguments must be positive and equal to '
             f'1 or a multiple of 3. nargs={nargs}'))
    if nargs == 1:
        result = np.zeros(init.end - init.start) + args[0]
    else:
        result = np.zeros(init.end - init.start) + avg
        for i in range(0, nargs, 3):
            result += o3gauss(*args[i:(i+3)])
    return result


def loglike(cube, ndim, nparams):
    cubeaslist = [cube[i] for i in range(ndim)]
    ymodel = model(*cubeaslist)
    loglikelihood = -0.5 * (((ymodel - ydata) / noise)**2).sum()
    return loglikelihood

#==============================================================================
# Define the output generating functions
#==============================================================================


# This write function is not yet universal
def write_result(ncomp, maxcomp, outmodel):
    with open(line_outfile, 'a') as f:
        entries = ([column, row, ncomp, avg, *outmodel]
                  + [0]*((maxcomp - ncomp) * 3))
        strtowrite = ('{:3s} {:3s} | {:1d} | {:7.2f}'
                      + maxcomp * ' | {:8.3f} {:6.3f} {:9.3e}'
                      + '\n')
        f.write(strtowrite.format(*entries))
    print(f'Best Fit: {ncomp} Components')

# still need to define the colorlist, which also needs to be able to cycle
# to allow for high number of components
colorlist = ['goldenrod', 'plum', 'teal', 'firebrick', 'darkorange']


def make_model_plot(ncomp, outmodel, loglike):
    fig, ax = plt.subplots()
    ax.set_xlim(init.plotmin, init.plotmax)
    ax.set_ylim(miny - ypadding, maxy + ypadding)
    ax.text(0.25, 0.95, f'ln(Z) = {loglike:.2f}', transform=ax.transAxes)
    ax.plot(x, ydata, '-', lw=1, color='0.75', label='data', zorder=1)
    # for 0 or 1 components, plot noise
    if (ncomp == 0) or (ncomp == 1):
        ax.plot(x, noise, '-', color='red', zorder=1)
    # plot horizontal dashed red line of the 3sigma error + the average
    ax.axhline(init.fluxsigma * stdev + avg, ls='--', lw=0.5, color='red',
               zorder=0)
    # Plot line of the expected location of the oiii line at the systemic
    # velocity of the system.
    ax.axvline(systemic, 0, 1, ls='--', lw=0.5, color='blue', zorder=0)
    # Plot the ranges from where the continuum was sampled.
    ax.axvspan(wave[init.low1], wave[init.upp1], facecolor='black', alpha=0.1)
    ax.axvspan(wave[init.low2], wave[init.upp2], facecolor='black', alpha=0.1)
    # Plot the best fit model.
    ax.plot(x, model(*outmodel), '-', lw=1, color='black', label='model',
            zorder=3)
    # Draw the components of the model if ncomp > 1
    if ncomp > 1:
        for i in range(0, 3*ncomp, 3):
            color = colorlist[(i//3)%(len(colorlist))]
            ax.plot(x, model(*outmodel[i:(i+3)]), '-', lw=0.75, color=color,
            zorder=2)
    ax.set_title(f'Pixel ({column:3s}, {row:3s}) -- {ncomp} Components')
    ax.set_xlabel('Wavelength ($\mathrm{\AA}$)')
    ax.set_ylabel('Flux ($\mathrm{10^{-20}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)')
    fig.savefig(plot_outfile + f'_{ncomp}_posterior.pdf')
    plt.close()


def remove_unnecessary_files(path, patterns):
    filestoremove = []
    for pattern in patterns:
        filelist = glob.glob(path + pattern)
        for file in filelist:
            filestoremove.append(file)
    for file in filestoremove:
        try:
            os.remove(file)
        except:
            print(f"Error while deleting '{file}'")

#==============================================================================
# Define the primary fitting routine
#==============================================================================


def mp_worker(index):
    # Set the variables that will be used directly within the functions below
    # rather than passing them in as arguments.
    global x, ydata, miny, maxy, avg, stdev, noise, wave, threesigma
    global column, row, line_outfile, ypadding, systemic, plot_outfile

    # Clean up the contents of the input directory
    if os.path.exists(os.path.join(specpath, '.DS_Store')):
        os.remove(os.path.join(specpath, '.DS_Store'))

    # Set the maximum number of components that this program will model
    maxcomp = 3

    # build lists that will hold the outputs for each model
    analyzers = [0] * (maxcomp + 1)
    lnZs = [0] * (maxcomp + 1)
    outmodels = [0] * (maxcomp + 1)

    # Set the filename and filepaths for the various inputs and outputs.
    infile = listing[index]
    infilebase = infile.split('.')[0]
    column = infilebase.split('_')[1] # spaxel column coordinate (x)
    row = infilebase.split('_')[2] # spaxel row coordinate (y)

    inspecpath = os.path.join(specpath, infile)
    # outspecpath = os.path.join(basepath, 'indata', init.target, 'donespec',
                               # infile)
    data_outfile = os.path.join(basepath, f'{target}_out', 'out',
                                f'{infilebase}_{init.line}')
    plot_outfile = os.path.join(basepath, f'{target}_out', 'plots',
                                f'{infilebase}_{init.line}')
    line_outfile = os.path.join(basepath, f'{target}_out',
                                f'{init.target}_{init.line}.txt')

    # Read in the data and start to cut it into the appropriate useful bits.
    wave, flux, noise = np.loadtxt(inspecpath, usecols=(0,1,2), unpack=True, delimiter=',')

    x = wave[init.start:init.end]
    ydata = flux[init.start:init.end]
    noise = noise[init.start:init.end]
    maxy = max(ydata)
    miny = min(ydata)
    ypadding = .05 * (maxy - miny)
    systemic = (1. + init.red) * init.orig_wave
    cont1 = flux[init.low1:init.upp1]
    cont2 = flux[init.low2:init.upp2]
    avg = (np.median(cont1) + np.median(cont2)) / 2
    ### should we re-evaluate the stdev of the continuum now that there is
    ### reported error?
    stdev = (np.std(cont1) + np.std(cont2)) / 2 # stnd dev of continuum flux
    threesigma = (3. * stdev) # * init.minwidth * np.sqrt(2*np.pi)
    # noise = stdev * np.sqrt((abs(ydata) / abs(avg))) # signal dependant noise

    #------ CONTINUUM FIT ------

    print(f'Pixel: {column}, {row}')
    print(f'Min Y: {miny}')
    print(f'Max Y: {maxy}')

    # Set the number of dimensions of this model
    ncomp = 0

    # parameters = ['contflux']
    # n_params = len(parameters)
    n_params = 1

    # run MultiNest
    pymultinest.run(loglike, prior, n_params,
                    outputfiles_basename=f'{data_outfile}_{ncomp}_',
                    n_live_points=200, multimodal=False, resume=False,
                    verbose=False)
    analyzers[ncomp] = pymultinest.Analyzer(
        outputfiles_basename=f'{data_outfile}_{ncomp}_',
        n_params=n_params)
    lnZs[ncomp] = analyzers[ncomp].get_stats()['global evidence']
    outmodels[ncomp] = analyzers[ncomp].get_best_fit()['parameters']

    # plot best fit
    make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp])

    # set this as the best fit
    bestncomp = ncomp

    # Now try increased number of components.
    for ncomp in range(1, maxcomp+1):
        n_params = 3 * ncomp
        print(f'Pixel: {column}, {row}')

        # run MultiNest
        pymultinest.run(loglike, prior, n_params,
                        outputfiles_basename=f'{data_outfile}_{ncomp}_',
                        n_live_points=200, multimodal=False, resume=False,
                        verbose=False)
        analyzers[ncomp] = pymultinest.Analyzer(
            outputfiles_basename=f'{data_outfile}_{ncomp}_',
            n_params=n_params)
        lnZs[ncomp] = analyzers[ncomp].get_stats()['global evidence']
        outmodels[ncomp] = analyzers[ncomp].get_best_fit()['parameters']

        # plot best fit
        make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp])

        if lnZs[ncomp] - lnZs[bestncomp] > 5.0:
            bestncomp = ncomp
        else:
            break

    write_result(bestncomp, maxcomp, outmodels[bestncomp])

    print(f'Average Continuum = {avg:.2f}')
    print(f'Standard deviation = {stdev:.4f}')

    # Move the completed spectrum to another directory
    # shutil.move(inspecpath, outspecpath)

    # Delete extraneous outfiles
    remove_unnecessary_files(data_outfile, ['*ev.dat', '*IS.*', '*live.points',
                                            '*phys_live.po', '*post_equal_w',
                                            '*resume.dat'])


def mp_handler():
    pool = mp.Pool(processes=3)
    pool.map(mp_worker, range(len(listing)))


def main():
    make_dirs()
    mp_handler()
    print('--- {0} seconds ---'.format(time.time() - start_time))


if __name__ == '__main__':
    main()
