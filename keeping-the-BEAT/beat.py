import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pymultinest
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
import glob

# TO DO change plot min and max back


class Fit(object):
    """ Fit gaussian component(s) to specified emission line(s) for a number of pixel/spaxel.
        Parameters
        ----------
        out_dir : str
            The path where to put output catalog and plots
        spec_dir : str
            Path to spectra text files
        load_file : function
            User defined function that takes the path to a spectra text file (of a single pixel/spaxel) and returns
            row, column, wave, flux, and noise. Wave, flux, noise are used for fitting while row and column are used
            for naming output data and files
        fit_instructions : dict
            Dictionary that contains information for line(s) to be fit, fitting preferences, and general target
            and plotting information
        target_param : dict
            Dictionary that contains information general target information, fitting preferences and plotting info
    """

    def __init__(self, out_dir, spec_dir, load_file, fit_instructions, target_param):
        self.basepath = out_dir
        self.specpath = spec_dir
        self.load_file = load_file
        self.fit_instructions = fit_instructions
        self.target_param = target_param
        self.cat = None

        self.listing = sorted(os.listdir(self.specpath))
        self.free_lines = sum(self.fit_instructions[line]['flux_free'] is True for line in self.fit_instructions)

        self.make_dirs()

    def make_dirs(self):
        """make directory structure for output data"""
        if not os.path.exists(os.path.join(self.basepath, f'{self.target_param["name"]}_out')):
            os.mkdir(os.path.join(self.basepath, f'{self.target_param["name"]}_out'))

        if not os.path.exists(os.path.join(self.basepath, f'{self.target_param["name"]}_out', 'out')):
            os.mkdir(os.path.join(self.basepath, f'{self.target_param["name"]}_out', 'out'))

        if not os.path.exists(os.path.join(self.basepath, f'{self.target_param["name"]}_out', 'plots')):
            os.mkdir(os.path.join(self.basepath, f'{self.target_param["name"]}_out', 'plots'))

        if not os.path.exists(os.path.join(self.basepath, f'{self.target_param["name"]}_out',
                                        f'{self.target_param["name"]}_{self.fit_instructions["line1"]["name"]}.txt')):
            self.init_cat()

    def make_gaussian(self, mu, sigma, flux):
        """ centroid, width, integrated flux"""
        h = flux / (sigma * np.sqrt(2 * np.pi))
        s = -1.0 / (2 * sigma ** 2)

        def f(x):
            return h * np.exp(s * (x - mu) ** 2)

        return f

    def gauss(self, pos1, width1, *args):
        """"""
        pos_all = []
        flux_all = []
        lines = []
        args = list(args)
        for line in self.fit_instructions:
            lines.append(line)  # keeping track of what lines have been fit
            pos = (pos1 - ((1 + self.target_param['red']) * (self.fit_instructions['line1']['wave']
                             - self.fit_instructions[line]['wave'])))
            pos_all.append(pos)
            if self.fit_instructions[line]['flux_free'] is True:
                flux = args[0]     # get the argument
                flux_all.append(flux)
                args.pop(0)
            else:
                line_lock = self.fit_instructions[line]['locked_with']
                ratio = self.fit_instructions[line]['flux_ratio']
                flux_lock = flux_all[lines.index(line_lock)]
                flux = flux_lock / ratio
                flux_all.append(flux)

        all_gauss_functions = [0] * len(x) # global variable at end
        for i in range(0, len(self.fit_instructions)):
            pos = pos_all[i]
            flux = flux_all[i]
            gauss_hold = self.make_gaussian(pos, width1, flux)
            all_gauss_functions += gauss_hold(x)
        return all_gauss_functions

    def prior(self, cube, ndim, nparams):
        if ((ndim != 1) and (ndim % (2 + self.free_lines) != 0)) or (ndim <= 0): # 3 or 2 + free_lines
            raise ValueError(
                ('The number of dimensions must be positive and equal to '
                 f'1 or a multiple of {2 + self.free_lines}. ndim={ndim}'))
        if ndim == 1:
            cube[0] = avg
        else:
            for i in range(0, ndim, 2 + self.free_lines):  # is this correct?? changed from 0,ndim,3
                # uniform wavelength prior
                cube[i + 0] = (self.fit_instructions['line1']['minwave'] + cube[i + 0]
                               * self.fit_instructions['line1']['wave_range'])
                # uniform width prior
                cube[i + 1] = (self.target_param['minwidth']
                               + cube[i + 1] * (self.target_param['maxwidth'] - self.target_param['minwidth']))
                # log-uniform flux prior
                for fprior in range(0, self.free_lines):
                    cube[i + fprior + 2] = (threesigma) * np.power(10, cube[i + fprior + 2] * 4) # MADE A CHANGE

                # need more flux priors depending on fluxes we have
                # number of flux priors is same as number of free lines

    def model(self, *args):
        nargs = len(args)
        if ((nargs != 1) and (nargs % (2 + self.free_lines) != 0)) or (nargs <= 0): # again is this 3 or 2 + free_lines
            raise ValueError(
                ('The number of arguments must be positive and equal to '
                 f'1 or a multiple of 3. nargs={nargs}'))
        if nargs == 1:
            result = np.zeros(self.target_param['end'] - self.target_param['start']) + args[0]
        else:
            result = np.zeros(self.target_param['end'] - self.target_param['start']) + avg
            for i in range(0, nargs, 2 + self.free_lines): # again is this 3 or 2 + free_lines
                result += self.gauss(*args[i:(i + (2 + self.free_lines))]) # 3 or 2 + free_lines
        return result

    def loglike(self, cube, ndim, nparams):
        cubeaslist = [cube[i] for i in range(ndim)]
        ymodel = self.model(*cubeaslist)
        loglikelihood = -0.5 * (((ymodel - ydata) / noise) ** 2).sum()
        return loglikelihood

    def write_results(self, index, ncomp, outmodel):
        cat_file = os.path.join(self.basepath, f'{self.target_param["name"]}_out', f'{self.target_param["name"]}_'
                                f'{self.fit_instructions["line1"]["name"]}.txt')
        cat = pd.read_csv(cat_file, index_col='index')
        use_col = cat.columns[2:len(outmodel) + 2].tolist()
        for i, mod in enumerate(outmodel):
            cat.at[index, use_col[i]] = mod
        cat.at[index, 'ncomps'] = ncomp
        cat.to_csv(cat_file, index_label='index')

    def make_model_plot(self, ncomp, outmodel, loglike):
        colorlist = ['goldenrod', 'plum', 'teal', 'firebrick', 'darkorange']
        fig, ax = plt.subplots()
        ax.set_xlim(self.target_param["plotmin"], self.target_param["plotmax"])
        ax.set_ylim(miny - ypadding, maxy + ypadding)
        ax.text(0.25, 0.95, f'ln(Z) = {loglike:.2f}', transform=ax.transAxes)
        ax.plot(x, ydata, '-', lw=1, color='0.75', label='data', zorder=1)

        # for 0 or 1 components, plot noise
        if (ncomp == 0) or (ncomp == 1):
            ax.plot(x, noise, '-', color='red', zorder=1)

        # plot horizontal dashed red line of the 3sigma error + the average
        ax.axhline(self.target_param["fluxsigma"] * stdev + avg, ls='--', lw=0.5, color='red',
                   zorder=0)

        # Plot line of the expected location of the oiii line at the systemic
        # velocity of the system.
        ax.axvline(systemic, 0, 1, ls='--', lw=0.5, color='blue', zorder=0)

        # Plot the ranges from where the continuum was sampled.
        # ax.axvspan(wave[self.low1], wave[self.upp1], facecolor='black', alpha=0.1)
        # ax.axvspan(wave[self.low2], wave[self.upp2], facecolor='black', alpha=0.1)
        ax.axvspan(self.target_param["continuum1"][0], self.target_param["continuum1"][1], facecolor='black', alpha=0.1)
        ax.axvspan(self.target_param["continuum2"][0], self.target_param["continuum2"][1], facecolor='black', alpha=0.1)

        # Plot the best fit model.
        ax.plot(x, self.model(*outmodel), '-', lw=1, color='black', label='model',
                zorder=3)

        # Draw the components of the model if ncomp > 1
        if ncomp > 1:
            for i in range(0, 3 * ncomp, 3):
                color = colorlist[(i // 3) % (len(colorlist))]
                ax.plot(x, self.model(*outmodel[i:(i + 2 + self.free_lines)]), '-', lw=0.75, color=color,
                        zorder=2)
        ax.set_title(f'Pixel ({column:3s}, {row:3s}) -- {ncomp} Components')
        ax.set_xlabel('Wavelength ($\mathrm{\AA}$)')
        ax.set_ylabel('Flux ($erg\ s^{-1}\ cm^{-2}\ \AA^{-1}}$)')
        fig.savefig(plot_outfile + f'_{ncomp}_posterior.pdf')
        plt.close()


    def init_cat(self):
        """Blank output catalog"""
        cols = ["index", "pixel", "ncomps"]
        for i in range(1, self.target_param["maxcomp"] + 1):
            cols.append(f'wave_{i}')
            cols.append(f'width_{i}')
            for flux in range(1, self.free_lines + 1):
                cols.append(f'flux_{i}_{chr(ord("@") + flux)}')
        self.cat = pd.DataFrame(np.zeros((len(self.listing), len(cols))),
                                columns=cols)
        self.cat.loc[:, 'ncomps'] = -1

        self.cat["index"] = self.cat.index

        outfile = os.path.join(self.basepath, f'{self.target_param["name"]}_out',
                               f'{self.target_param["name"]}_{self.fit_instructions["line1"]["name"]}.txt')

        self.cat['pixel'] = self.cat.apply(self.get_pix, axis=1)

        self.cat.to_csv(outfile, index=False)

    def get_pix(self, row_ind):
        fil = self.listing[int(row_ind['index'])]
        # print(f'fil:{fil}')
        infilebase = fil.split('.')[0]
        # print(f'infilebase:{infilebase}')
        column = infilebase.split('_')[1]  # spaxel column coordinate (x)
        row = infilebase.split('_')[2]  # spaxel row coordinate (y)
        return f'{column}_{row}'

    def find_unfit(self):
        open_cat = pd.read_csv(os.path.join(self.basepath, f'{self.target_param["name"]}_out',
                                    f'{self.target_param["name"]}_{self.fit_instructions["line1"]["name"]}.txt'))
        indices = list(np.where(open_cat["ncomps"]==-1)[0])
        return indices

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

    def mp_worker(self, index):
        # Set the variables that will be used directly within the functions below
        # rather than passing them in as arguments.
        global x, ydata, miny, maxy, avg, stdev, noise, wave, threesigma
        global column, row, line_outfile, ypadding, systemic, plot_outfile

        # Clean up the contents of the input directory
        if os.path.exists(os.path.join(self.specpath, '.DS_Store')):
            os.remove(os.path.join(self.specpath, '.DS_Store'))

        # Set the filename and filepaths for the various inputs and outputs.
        infile = self.listing[index]
        inspecpath = os.path.join(self.specpath, infile)
        row, column, wave, flux, noise = self.load_file(inspecpath)
        infilebase = infile.split('.')[0]
        data_outfile = os.path.join(self.basepath, f'{self.target_param["name"]}_out', 'out',
                                    f'{infilebase}_{self.fit_instructions["line1"]["name"]}')
        plot_outfile = os.path.join(self.basepath, f'{self.target_param["name"]}_out', 'plots',
                                    f'{infilebase}_{self.fit_instructions["line1"]["name"]}')
        line_outfile = os.path.join(self.basepath, f'{self.target_param["name"]}_out',
                                    f'{self.target_param["name"]}_{self.fit_instructions["line1"]["name"]}.txt')

        x = wave[self.target_param["start"]:self.target_param["end"]]
        ydata = flux[self.target_param["start"]:self.target_param["end"]]
        noise = noise[self.target_param["start"]:self.target_param["end"]]
        maxy = max(ydata)
        miny = min(ydata)
        ypadding = .05 * (maxy - miny)

        # is this right? should this be for all lines??
        systemic = (1. + self.target_param["red"]) * self.fit_instructions["line1"]["wave"]

        low1_ind = (np.abs(wave - self.target_param["continuum1"][0])).argmin()
        upp1_ind = (np.abs(wave - self.target_param["continuum1"][1])).argmin()
        low2_ind = (np.abs(wave - self.target_param["continuum2"][0])).argmin()
        upp2_ind = (np.abs(wave - self.target_param["continuum2"][1])).argmin()
        cont1 = flux[low1_ind:upp1_ind]
        cont2 = flux[low2_ind:upp2_ind]

        avg = (np.median(cont1) + np.median(cont2)) / 2
        ### should we re-evaluate the stdev of the continuum now that there is
        ### reported error?
        stdev = (np.std(cont1) + np.std(cont2)) / 2  # stnd dev of continuum flux
        threesigma = (3. * stdev)  # * init.minwidth * np.sqrt(2*np.pi)
        # noise = stdev * np.sqrt((abs(ydata) / abs(avg))) # signal dependant noise

        # Set the maximum number of components that this program will model
        maxcomp = self.target_param["maxcomp"]

        # build lists that will hold the outputs for each model
        analyzers = [0] * (maxcomp + 1)
        lnZs = [0] * (maxcomp + 1)
        outmodels = [0] * (maxcomp + 1)

        # ------ CONTINUUM FIT ------

        print(f'Fitting Pixel: {column}, {row}')
        print(f'Min Y: {miny:.2e}')
        print(f'Max Y: {maxy:.2e}')

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
            n_params=n_params, verbose=False)
        lnZs[ncomp] = analyzers[ncomp].get_stats()['global evidence']
        outmodels[ncomp] = analyzers[ncomp].get_best_fit()['parameters']

        # plot best fit
        self.make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp])

        # set this as the best fit
        bestncomp = ncomp

        # Now try increased number of components.
        for ncomp in range(1, maxcomp + 1):
            print(f'Pixel: {column}, {row} trying {ncomp} component(s)')
            n_params = 3 * ncomp

            # run MultiNest
            pymultinest.run(self.loglike, self.prior, n_params,
                            outputfiles_basename=f'{data_outfile}_{ncomp}_',
                            n_live_points=200, multimodal=False, resume=False,
                            verbose=False)
            analyzers[ncomp] = pymultinest.Analyzer(
                outputfiles_basename=f'{data_outfile}_{ncomp}_',
                n_params=n_params, verbose=False)
            lnZs[ncomp] = analyzers[ncomp].get_stats()['global evidence']
            outmodels[ncomp] = analyzers[ncomp].get_best_fit()['parameters']

            # plot best fit
            self.make_model_plot(ncomp, outmodels[ncomp], lnZs[ncomp])

            if lnZs[ncomp] - lnZs[bestncomp] > self.target_param["lnz"]:
                bestncomp = ncomp
            else:
                break

        self.write_results(index, bestncomp, outmodels[bestncomp])

        print(f'Pixel: {column}, {row} fit with {bestncomp} components')
        print(f'Average Continuum = {avg:.2e}')
        print(f'Standard deviation = {stdev:.4e}')


        # Delete extraneous outfiles
        self.remove_unnecessary_files(data_outfile, ['*ev.dat', '*IS.*', '*live.points',
                                                     '*phys_live.po', '*post_equal_w',
                                                     '*resume.dat'])

    def mp_handler(self):
        # make mp optional
        unfit_pix = self.find_unfit()
        print(f'remaining to fit: {len(unfit_pix)}')
        pool = mp.Pool(processes=self.target_param["cores"])
        pool.map(self.mp_worker, unfit_pix)
