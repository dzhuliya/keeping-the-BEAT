# import init_oiii as init
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
#start_time = time.time()


class Fit(object):

    def __init__(self, out_dir, spec_dir, target, red, minwave, wave_range, minwidth, maxwidth, start, end, plotmin,
                 plotmax, fluxsigma, low1, low2, upp1, upp2, line, orig_wave):
        # Get a listing of the spectra involved in this run.
        self.basepath = out_dir
        self.specpath = spec_dir
        self.target = target
        self.red = red
        self.minwave = minwave
        self.wave_range = wave_range
        self.minwidth = minwidth
        self.maxwidth = maxwidth
        self.start = start
        self.end = end
        self.plotmin = plotmin
        self.plotmax = plotmax
        self.fluxsigma = fluxsigma
        self.low1 = low1
        self.low2 = low2
        self.upp1 = upp1
        self.upp2 = upp2
        self.line = line
        self.orig_wave = orig_wave

        self.listing = sorted(os.listdir(self.specpath))

        # Provide some names and values for the lines that are being fit here.
        self.lines = {
            'H-beta': 4861.3,
            '[O III] 4959': 4958.92,
            '[O III] 5007': 5006.84}
        self.make_dirs()

    def make_dirs(self):
        """make directory structure for output data"""
        print('making_dirs')
        if not os.path.exists(os.path.join(self.basepath, f'{self.target}_out')):
            os.mkdir(os.path.join(self.basepath, f'{self.target}_out'))

        if not os.path.exists(os.path.join(self.basepath, f'{self.target}_out', 'out')):
            os.mkdir(os.path.join(self.basepath, f'{self.target}_out', 'out'))

        if not os.path.exists(os.path.join(self.basepath, f'{self.target}_out', 'plots')):
            os.mkdir(os.path.join(self.basepath, f'{self.target}_out', 'plots'))

    # ==============================================================================
    # Define the Gaussian models.
    # ==============================================================================

    def make_gaussian(self, mu, sigma, flux):
        h = flux / (sigma * np.sqrt(2 * np.pi))
        s = -1.0 / (2 * sigma ** 2)

        def f(x):
            return h * np.exp(s * (x - mu) ** 2)

        return f

    def o3gauss(self, pos1, width1, flux1):
        # [O III] is a doublet with a relationship between the two lines.
        # Define the second Gaussian in terms of the first.
        pos2 = (pos1 - ((1 + self.red)
                        * (self.lines['[O III] 5007'] - self.lines['[O III] 4959'])))
        width2 = width1
        flux2 = flux1 / 3
        # make and return the combined Gaussian
        gauss1 = self.make_gaussian(pos1, width1, flux1)
        gauss2 = self.make_gaussian(pos2, width2, flux2)
        return gauss1(x) + gauss2(x)

    # ==============================================================================
    # Define prior, model, and loglikelihood functions that work for any number
    # of components.
    # ==============================================================================

    def prior(self, cube, ndim, nparams):
        if ((ndim != 1) and (ndim % 3 != 0)) or (ndim <= 0):
            raise ValueError(
                ('The number of dimensions must be positive and equal to '
                 f'1 or a multiple of 3. ndim={ndim}'))
        if ndim == 1:
            cube[0] = avg
        else:
            for i in range(0, ndim, 3):
                # uniform wavelength prior
                cube[i + 0] = (self.minwave + cube[i + 0]
                               * self.wave_range)
                # uniform width prior
                cube[i + 1] = (self.minwidth
                               + cube[i + 1] * (self.maxwidth - self.minwidth))
                # log-uniform flux prior
                cube[i + 2] = (threesigma * cube[i + 1] * np.sqrt(2 * np.pi)
                               * np.power(10, cube[i + 2] * 4))

    def model(self, *args):
        nargs = len(args)
        if ((nargs != 1) and (nargs % 3 != 0)) or (nargs <= 0):
            raise ValueError(
                ('The number of arguments must be positive and equal to '
                 f'1 or a multiple of 3. nargs={nargs}'))
        if nargs == 1:
            result = np.zeros(self.end - self.start) + args[0]
        else:
            result = np.zeros(self.end - self.start) + avg
            for i in range(0, nargs, 3):
                result += self.o3gauss(*args[i:(i + 3)])
        return result

    def loglike(self, cube, ndim, nparams):
        cubeaslist = [cube[i] for i in range(ndim)]
        ymodel = self.model(*cubeaslist)
        loglikelihood = -0.5 * (((ymodel - ydata) / noise) ** 2).sum()
        return loglikelihood

    # ==============================================================================
    # Define the output generating functions
    # ==============================================================================

    # This write function is not yet universal
    def write_result(self, ncomp, maxcomp, outmodel):
        with open(line_outfile, 'a') as f:
            entries = ([column, row, ncomp, avg, *outmodel]
                       + [0] * ((maxcomp - ncomp) * 3))
            strtowrite = ('{:3s} {:3s} | {:1d} | {:7.2f}'
                          + maxcomp * ' | {:8.3f} {:6.3f} {:9.3e}'
                          + '\n')
            f.write(strtowrite.format(*entries))
        print(f'Best Fit: {ncomp} Components')

    # still need to define the colorlist, which also needs to be able to cycle
    # to allow for high number of components
    colorlist = ['goldenrod', 'plum', 'teal', 'firebrick', 'darkorange']

    def make_model_plot(self, ncomp, outmodel, loglike):
        fig, ax = plt.subplots()
        ax.set_xlim(self.plotmin, self.plotmax)
        ax.set_ylim(miny - ypadding, maxy + ypadding)
        ax.text(0.25, 0.95, f'ln(Z) = {loglike:.2f}', transform=ax.transAxes)
        ax.plot(x, ydata, '-', lw=1, color='0.75', label='data', zorder=1)
        # for 0 or 1 components, plot noise
        if (ncomp == 0) or (ncomp == 1):
            ax.plot(x, noise, '-', color='red', zorder=1)
        # plot horizontal dashed red line of the 3sigma error + the average
        ax.axhline(self.fluxsigma * stdev + avg, ls='--', lw=0.5, color='red',
                   zorder=0)
        # Plot line of the expected location of the oiii line at the systemic
        # velocity of the system.
        ax.axvline(systemic, 0, 1, ls='--', lw=0.5, color='blue', zorder=0)
        # Plot the ranges from where the continuum was sampled.
        ax.axvspan(wave[self.low1], wave[self.upp1], facecolor='black', alpha=0.1)
        ax.axvspan(wave[self.low2], wave[self.upp2], facecolor='black', alpha=0.1)
        # Plot the best fit model.
        ax.plot(x, self.model(*outmodel), '-', lw=1, color='black', label='model',
                zorder=3)
        # Draw the components of the model if ncomp > 1
        if ncomp > 1:
            for i in range(0, 3 * ncomp, 3):
                color = self.colorlist[(i // 3) % (len(self.colorlist))]
                ax.plot(x, self.model(*outmodel[i:(i + 3)]), '-', lw=0.75, color=color,
                        zorder=2)
        ax.set_title(f'Pixel ({column:3s}, {row:3s}) -- {ncomp} Components')
        ax.set_xlabel('Wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel('Flux ($\mathrm{10^{-20}\ erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)')
        fig.savefig(plot_outfile + f'_{ncomp}_posterior.pdf')
        plt.close()

    def remove_unnecessary_files(self, path, patterns):
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

    # ==============================================================================
    # Define the primary fitting routine
    # ==============================================================================

    def mp_worker(self, index):
        # Set the variables that will be used directly within the functions below
        # rather than passing them in as arguments.
        global x, ydata, miny, maxy, avg, stdev, noise, wave, threesigma
        global column, row, line_outfile, ypadding, systemic, plot_outfile

        # Clean up the contents of the input directory
        if os.path.exists(os.path.join(self.specpath, '.DS_Store')):
            os.remove(os.path.join(self.specpath, '.DS_Store'))

        # Set the maximum number of components that this program will model
        maxcomp = 3

        # build lists that will hold the outputs for each model
        analyzers = [0] * (maxcomp + 1)
        lnZs = [0] * (maxcomp + 1)
        outmodels = [0] * (maxcomp + 1)

        # Set the filename and filepaths for the various inputs and outputs.
        infile = self.listing[index]
        infilebase = infile.split('.')[0]
        column = infilebase.split('_')[1]  # spaxel column coordinate (x)
        row = infilebase.split('_')[2]  # spaxel row coordinate (y)

        inspecpath = os.path.join(self.specpath, infile)
        # outspecpath = os.path.join(basepath, 'indata', init.target, 'donespec',
        # infile)
        data_outfile = os.path.join(self.basepath, f'{self.target}_out', 'out',
                                    f'{infilebase}_{self.line}')
        plot_outfile = os.path.join(self.basepath, f'{self.target}_out', 'plots',
                                    f'{infilebase}_{self.line}')
        line_outfile = os.path.join(self.basepath, f'{self.target}_out',
                                    f'{self.target}_{self.line}.txt')

        # Read in the data and start to cut it into the appropriate useful bits.
        wave, flux, noise = np.loadtxt(inspecpath, usecols=(0, 1, 2), unpack=True, delimiter=',')

        x = wave[self.start:self.end]
        ydata = flux[self.start:self.end]
        noise = noise[self.start:self.end]
        maxy = max(ydata)
        miny = min(ydata)
        ypadding = .05 * (maxy - miny)
        systemic = (1. + self.red) * self.orig_wave
        cont1 = flux[self.low1:self.upp1]
        cont2 = flux[self.low2:self.upp2]
        avg = (np.median(cont1) + np.median(cont2)) / 2
        ### should we re-evaluate the stdev of the continuum now that there is
        ### reported error?
        stdev = (np.std(cont1) + np.std(cont2)) / 2  # stnd dev of continuum flux
        threesigma = (3. * stdev)  # * init.minwidth * np.sqrt(2*np.pi)
        # noise = stdev * np.sqrt((abs(ydata) / abs(avg))) # signal dependant noise

        # ------ CONTINUUM FIT ------

        print(f'Pixel: {column}, {row}')
        print(f'Min Y: {miny}')
        print(f'Max Y: {maxy}')

        # Set the number of dimensions of this model
        ncomp = 0

        # parameters = ['contflux']
        # n_params = len(parameters)
        n_params = 1

        # run MultiNest
        pymultinest.run(self.loglike, self.prior, n_params,
                        outputfiles_basename=f'{data_outfile}_{ncomp}_',
                        n_live_points=200, multimodal=False, resume=False,
                        verbose=False)
        analyzers[ncomp] = pymultinest.Analyzer(
            outputfiles_basename=f'{data_outfile}_{ncomp}_',
            n_params=n_params)
        lnZs[ncomp] = analyzers[ncomp].get_stats()['global evidence']
        outmodels[ncomp] = analyzers[ncomp].get_best_fit()['parameters']

        # plot best fit
        self.make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp])

        # set this as the best fit
        bestncomp = ncomp

        # Now try increased number of components.
        for ncomp in range(1, maxcomp + 1):
            n_params = 3 * ncomp
            print(f'Pixel: {column}, {row}')

            # run MultiNest
            pymultinest.run(self.loglike, self.prior, n_params,
                            outputfiles_basename=f'{data_outfile}_{ncomp}_',
                            n_live_points=200, multimodal=False, resume=False,
                            verbose=False)
            analyzers[ncomp] = pymultinest.Analyzer(
                outputfiles_basename=f'{data_outfile}_{ncomp}_',
                n_params=n_params)
            lnZs[ncomp] = analyzers[ncomp].get_stats()['global evidence']
            outmodels[ncomp] = analyzers[ncomp].get_best_fit()['parameters']

            # plot best fit
            self.make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp])

            if lnZs[ncomp] - lnZs[bestncomp] > 5.0:
                bestncomp = ncomp
            else:
                break

        self.write_result(bestncomp, maxcomp, outmodels[bestncomp])

        print(f'Average Continuum = {avg:.2f}')
        print(f'Standard deviation = {stdev:.4f}')

        # Move the completed spectrum to another directory
        # shutil.move(inspecpath, outspecpath)

        # Delete extraneous outfiles
        self.remove_unnecessary_files(data_outfile, ['*ev.dat', '*IS.*', '*live.points',
                                                     '*phys_live.po', '*post_equal_w',
                                                     '*resume.dat'])

    def mp_handler(self):
        pool = mp.Pool(processes=3)
        pool.map(self.mp_worker, range(len(self.listing)))
