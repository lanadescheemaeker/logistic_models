import matplotlib as mpl
import make_colormap as mc
import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec

import sys
sys.path.insert(1, '../sglv_timeseries')

import sglv_timeseries.glv.Timeseries

from matplotlib.colors import Normalize
from make_colormap import *

import pandas as pd

import noise_analysis
from scipy import signal, stats

from timeseries_plotting import PlotTimeseries
from noise_analysis import noise_color
from neutrality_analysis import KullbackLeibler_neutrality
from neutral_covariance_test import neutral_covariance_test

from smooth_spline import *

# colormap noise

c = mpl.colors.ColorConverter().to_rgb
noise_cmap = make_colormap(
    [c('k'), c('brown'), 0.33, c('brown'), c('pink'), 0.66, c('pink'), c('lightgrey')])  # with grey
noise_lim = [-3, 0]
noise_cmap_ww = make_colormap(
    [c('k'), c('brown'), 0.33, c('brown'), c('pink'), 0.66, c('pink'), c('white')])  # with white


# code from https://stackoverflow.com/questions/30465080/associating-colors-from-a-continuous-colormap-to-specific-values-in-matplotlib
class PiecewiseNormalize(Normalize):
    def __init__(self, xvalues, cvalues):
        self.xvalues = xvalues
        self.cvalues = cvalues

        Normalize.__init__(self)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        if self.xvalues is not None:
            x, y = self.xvalues, self.cvalues
            return np.ma.masked_array(np.interp(value, x, y))
        else:
            return Normalize.__call__(self, value, clip)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def example_noise_fit(ax, ts, label=None, verbose=False, spline=False, linear_all=False):
    frq, f = signal.periodogram(ts)

    frq = frq.astype(float)
    if np.any(np.imag(f) != 0):
        raise UserError('One of the densities is complex, check what went wrong.')
    else:
        r = np.array([ff.real for ff in f])  # TODO strange error np.real(f) and f.real do not behave as expected
        f = r.astype(float)

    mask = np.isfinite(f) & np.isfinite(frq)
    frq = frq[mask]
    f = f[mask]

    mask = (f > 0) & (frq > 0)
    frq = frq[mask]
    f = f[mask]

    frq = np.log10(frq)
    f = np.log10(np.abs(f))

    # plot points
    l = ax.plot(frq, f, '.', label=label, markersize=2, alpha=0.25)

    if len(frq) > 5:
        if spline:
            # spline interpolation
            p_spline = get_natural_cubic_spline_model(frq, f, minval=min(frq), maxval=max(frq), n_knots=4)

            y = p_spline.predict(frq)

            deriv = (y[1:] - y[:-1]) / (frq[1:] - frq[:-1])

            # plot spline interpolation
            ax.plot(frq, y, color=mc.change_color(l[0].get_color(), 1),
                    linestyle='dotted')  # , label = 'spline fit: %.2f' % min(deriv))

        if linear_all:
            x = np.linspace(min(frq), max(frq), 200)

            slope_all, intercept, r_value, p_value, std_err = stats.linregress(frq, f)

            if verbose:
                print("The slope with all points included is %.3f +- %.3f" % (slope_all, std_err))

            # plot linear interpolation
            ax.plot(x, slope_all * x + intercept, color=mc.change_color(l[0].get_color(), 0.8), linestyle='dashed')

        # only consider frequencies which correspond to periods that are smaller than (length_timeseries/10)
        # otherwise effects from windowing
        f = f[frq >= min(frq) + 1]
        frq = frq[frq >= min(frq) + 1]

        x = np.linspace(min(frq), max(frq), 200)

        slope, intercept, r_value, p_value, std_err = stats.linregress(frq, f)

        if verbose:
            print("The slope is %.3f +- %.3f" % (slope, std_err))

        # plot linear interpolation
        ax.plot(x, slope * x + intercept, color=mc.change_color(l[0].get_color(), 1.2),
                label='%.2f' % slope if not spline else "%.2f | %.2f | %.2f" % (slope, slope_all, min(deriv)))

        # spline interpolation without low frequencies
        # p_spline = get_natural_cubic_spline_model(frq, f, minval=min(frq), maxval=max(frq), n_knots=3.5)

        if spline:
            y = p_spline.predict(frq)

            # plot new spline interpolation
            # plt.plot(frq, y, color=change_color(l[0].get_color(), 1.3), label='spline 2')
    else:
        slope = np.nan

    ax.set_xlabel('log$_{10}$(frequency)')
    ax.set_ylabel('log$_{10}$(power spectral density)')

    return slope


class PlotCharacteristics():
    def __init__(self, ts, species=None):
        self.ts = ts

        self.mean = ts.mean()
        self.mean.drop('time', inplace=True)

        self.vmin = 0.1 * np.nanmin(self.mean.values[self.mean.values != np.inf])
        self.vmax = 10 * np.nanmax(self.mean.values[self.mean.values != np.inf])

        self.Nspecies = len(self.ts.columns) - 1

        self.noise_color = None

        if species == None:
            self.selection = self.select_species()
        else:
            self.selection = species

    def select_species(self):
        sorted_species = self.mean.sort_values().index.tolist()[::-1]

        return sorted_species[::max(1, int(self.Nspecies / 4))]

    def plot_timeseries(self, ax, species=None, raw=False):
        PlotTimeseries(self.ts, ax, species, raw)

    def plot_power_spectral_density(self, ax, species=None, mean_slope=False, raw=False):
        if len(self.ts) < 2:
            return

        if species != None:
            self.selection = species

        for s in self.selection:
            example_noise_fit(ax, self.ts[s])

        if mean_slope:
            if self.noise_color == None:
                self.noise_color = noise_analysis.noise_color(self.ts)

            ax.legend([], [], title='mean slope = %.2f + %.2f' % (
            np.mean(self.noise_color['slope_linear']), np.std(self.noise_color['slope_linear'])))

        if raw:
            ax.set_ylabel('')
            ax.set_xlabel('')

    def plot_noise_color(self, ax, raw=False):
        if len(self.ts) < 2:
            return

        if self.noise_color == None:
            self.noise_color = noise_color(self.ts)

        ax.scatter(self.mean, self.noise_color['slope_linear'])
        ax.errorbar(self.mean, self.noise_color['slope_linear'], self.noise_color['std_slope_linear'], linestyle='')

        xx = np.linspace(2, -3, 500).reshape([500, 1])
        ax.imshow(xx, cmap=noise_cmap_ww, vmin=noise_lim[0], vmax=noise_lim[1], extent=(self.vmin, self.vmax, -3, 2),
                  aspect='auto', alpha=0.75)

        if not raw:
            ax.set_ylabel('Slope power spectral density')

    def plot_absolute_step(self, ax, raw=False):
        if len(self.ts) < 2:
            return

        dx = (self.ts.values[1:, 1:].astype(float) - self.ts.values[:-1, 1:].astype(float))  # / x.values[:-1, 1:];
        dx[~np.isfinite(dx)] = np.nan

        if np.any(~np.isnan(dx)):
            mean_dx = np.nanmean(abs(dx), axis=0)
        else:
            return

        x = np.log10(self.mean[~np.isnan(mean_dx)])
        y = np.log10(mean_dx[~np.isnan(mean_dx)])

        if len(x) > 0:
            p_lin = np.polyfit(x, y, deg=1, cov=False)
        else:
            p_lin = np.nan, np.nan

        xx = [np.nanmin(self.mean.values), np.nanmax(self.mean.values)]
        ax.plot(xx, 10 ** (p_lin[1] + p_lin[0] * np.log10(xx)), c='k', linewidth=0.5)
        ax.text(0.95, 0.05, r'y $\propto$ x$^{%.2f}$' % p_lin[0], transform=ax.transAxes, va='bottom', ha='right')

        ax.scatter(self.mean, mean_dx)

        if not raw:
            ax.set_ylabel(r'$\langle \vert x(t+\delta t) - x(t)\vert \rangle$')

    def plot_width_distribution_ratios(self, ax, raw=False):
        if len(self.ts) < 2:
            return

        def fit_ratio(x):
            x = [x1 / x2 for x1, x2 in zip(x[:-1], x[1:]) if x1 != 0 and x2 != 0 and ~np.isnan(x1) and ~np.isnan(x2)]

            if len(x) > 5:
                a, b, c = stats.lognorm.fit(x, floc=0)  # Gives the paramters of the fit
                stat, pval = stats.kstest(x, 'lognorm', args=((a, b, c)))

                return a, b, c, stat, pval
            else:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)

        dx_ratio = pd.DataFrame(index=self.ts.columns, columns=['s', 'loc', 'scale', 'ks-stat', 'ks-pval'])
        dx_ratio.drop('time', inplace=True)

        for idx in dx_ratio.index:
            dx_ratio.loc[idx] = fit_ratio(self.ts[idx].values)  # b = 0, c = 1

        ax.scatter(self.mean, dx_ratio['s'], c=dx_ratio['ks-pval'], vmin=0, vmax=1, cmap='coolwarm')

    def plot_rank_abundance(self, ax, selected_times=None, raw=False):
        if selected_times == None:
            selected_times = self.ts['time'][::max(1, int(len(self.ts['time']) / 3))]

        for t in selected_times:
            abundance_profile = self.ts[self.ts['time'] == t].values.flatten()[1:]
            ax.plot(range(1, len(abundance_profile) + 1), np.sort(abundance_profile)[::-1], label='Day %d' % int(t))

        if not raw:
            ax.set_ylabel('Abundance')

    def plot_neutrality_measures(self, ax_KL, ax_NCT, raw=False):
        if len(self.ts) < 2:
            return

        KL = KullbackLeibler_neutrality(self.ts)
        norm_ts = self.ts.values[:, 1:].copy().astype(float)
        norm_ts /= norm_ts.sum(axis=1, keepdims=True)
        NCT = neutral_covariance_test(norm_ts, ntests=500, method='Kolmogorov', seed=56)

        ax_KL.matshow([[np.log10(KL)]], cmap='Blues_r', vmin=-1, vmax=3, aspect='auto', )
        ax_KL.set_xticks([])
        ax_KL.set_yticks([0])
        ax_KL.set_yticklabels([r'$D_{KL}$'], fontsize=10)
        ax_KL.text(0, 0, '{:0.2E}'.format(KL), ha='center', va='center', color='w' if KL < 10 ** (0.5) else 'k')

        norm = PiecewiseNormalize([self.vmin, np.log10(0.05), self.vmax], [0, 0.5, 1])
        ax_NCT.matshow([[np.log10(NCT)]], norm=norm, cmap='seismic_r', aspect='auto', vmin=-5, vmax=0)
        ax_NCT.set_xticks([])
        ax_NCT.set_yticks([0])
        ax_NCT.set_yticklabels([r'$p_{NCT}$'], fontsize=10)
        ax_NCT.text(0, 0, '{:0.2E}'.format(NCT), ha='center', va='center',
                    color='w' if NCT < 10 ** (-3) or NCT > 10 ** (-0.7) else 'k')


class PlotTimeseriesComparison():
    def __init__(self, files, titles=[], composition=['ts', 'psd', 'nc', 'dx', 'disdx', 'ra', 'nn'], vertical=True,
                 fig=None):
        if isinstance(files, str):
            self.files = np.array([pd.read_csv(files, na_values='NAN', index_col=0)])
        elif isinstance(files, pd.DataFrame):
            files = np.array([files])
        elif isinstance(files, list):
            if all(isinstance(file, str) for file in files):
                self.files = [pd.read_csv(file, na_values='NAN') for file in files]
            elif all(isinstance(file, pd.DataFrame) for file in files):
                self.files = files
            else:
                types = [type(file) for file in files]
                raise ValueError(
                    "All files should be of type str or pd.DataFrame, these files are of type: %s" % str(types))
        else:
            raise ValueError("All files should be of type str or pd.DataFrame, this file is of type %s" % type(files))

        for i, file in enumerate(self.files):
            mask = file[[col for col in file.columns if col.startswith('species_')]].dropna(how='all',
                                                                                            axis='index').index
            self.files[i] = file.loc[mask]

        # define figure
        if fig == None:
            if vertical == True:
                self.fig = plt.figure(figsize=(3 * len(files), 2.5 * len(composition)))  # , tight_layout=True)
            else:
                self.fig = plt.figure(figsize=(3 * len(composition), 2.5 * len(files)))  # , tight_layout=True)
        elif isinstance(fig, matplotlib.axes.Axes) and len(composition) == 1 and len(files) == 1:
            self.fig = None
        elif fig != 0 and composition == ['nn'] and len(fig) == 2:
            self.fig = None
        else:
            self.fig = fig

        # define titles
        if len(files) != len(titles):
            self.titles = ['' for _ in range(len(files))]
        else:
            self.titles = titles

        self.composition = composition

        if vertical:
            self.orientation = 'vertical'
        else:
            self.orientation = 'horizontal'

        # define grid
        self.set_grid_subfigures()

        self.axes = {'ts': [], 'psd': [], 'nc': [], 'dx': [], 'disdx': [], 'ra': [], 'KL': [], 'NCT': []}

        # define all axes
        if isinstance(fig, matplotlib.axes.Axes) and len(composition) == 1 and len(files) == 1:
            self.axes[composition[0]] = [fig]
        elif fig != 0 and composition == ['nn'] and len(fig) == 2:
            self.axes['KL'] = [fig[0]]
            self.axes['NCT'] = [fig[1]]
        else:
            self.define_axes()

        # draw all
        for i, file, title in zip(range(len(files)), self.files, self.titles):
            self.draw_time_series(i, file, title)

        # set x- and y-labels
        self.set_labels()

        # set scales and grid
        self.set_scales_axes()

        # remove ticklabels of shared axes
        if self.orientation == 'vertical' and len(self.files) > 0:
            for c in composition:
                if c != 'nn':
                    for ax in self.axes[c][1:]:
                        ax.tick_params(axis="both", left=True, labelleft=False)

        # limit visible yrange of timeseries (do not show values that go to values close to zero/infinity)
        if 'ts' in composition:
            ylim1, ylim2 = self.axes['ts'][0].get_ylim()
            ylim1 = max(1e-5, ylim1)
            ylim2 = min(1e6, ylim2)
            self.axes['ts'][0].set_ylim([ylim1, ylim2])

    def set_grid_subfigures(self):
        if self.orientation == 'vertical':
            self.gs = gridspec.GridSpec(len(self.composition), len(self.files), top=0.9, bottom=0.2, wspace=0.1,
                                        hspace=0.5, left=0.1, right=0.9)
        else:
            self.gs = gridspec.GridSpec(len(self.files), len(self.composition), top=0.9, bottom=0.2, wspace=0.5,
                                        width_ratios=[2 if ci == 'nn' else 3 for ci in self.composition], left=0.1,
                                        right=0.9)

    def set_labels(self):
        for c, xlabel, ylabel in zip(['ts', 'psd', 'nc', 'dx', 'disdx', 'ra'],
                                     ['Time', 'log$_{10}$(frequency)', 'Mean abundance', 'Mean abundance',
                                      'Mean abundance', 'Rank'], ['Abundance', 'log$_{10}$(power spectral density)',
                                                                  'Slope power \n spectral density',
                                                                  r'$\langle \vert x(t+\delta t) - x(t) \vert \rangle$',
                                                                  'Width distribution ratios \n of successive time points',
                                                                  'Abundance']):
            if c in self.composition:
                self.axes[c][0].set_ylabel(ylabel)
                self.axes[c][-1].set_xlabel(xlabel, x=1, ha='right')

    def define_axes(self):
        for i in range(len(self.files)):
            for c in self.composition:
                if self.orientation == 'vertical':
                    row = self.composition.index(c)
                    col = i
                else:
                    col = self.composition.index(c)
                    row = i

                if c == 'nn':
                    sub_gs = self.gs[row, col].subgridspec(4, 1, height_ratios=[2, 1, 1, 2])
                    self.axes['KL'] += [self.fig.add_subplot(sub_gs[1])]
                    self.axes['NCT'] += [self.fig.add_subplot(sub_gs[2])]
                else:
                    self.axes[c] += [self.fig.add_subplot(self.gs[row, col], sharey=self.axes[c][0] if i > 0 else None)]

    def set_scales_axes(self):
        for c, xscale, yscale, grid in zip(['ts', 'psd', 'nc', 'dx', 'disdx', 'ra'],
                                           ['linear', 'linear', 'log', 'log', 'log', 'log'],
                                           ['log', 'linear', 'linear', 'log', 'log', 'log'],
                                           [True, True, True, True, True, True]):

            if c in self.composition:
                for ax in self.axes[c]:
                    ax.set_yscale(yscale)
                    ax.set_xscale(xscale)
                    ax.grid(grid)

    def draw_time_series(self, i, file, title):
        if isinstance(file, str):
            ts = pd.read_csv(file, na_values='NAN')
        elif isinstance(file, pd.DataFrame):
            ts = file.copy()

        # set title
        if self.composition[0] != 'nn':
            self.axes[self.composition[0]][i].set_title(title)
        else:
            self.axes['KL'][i].set_title(title)

        plotter = PlotCharacteristics(ts)

        for c, func in zip(['ts', 'psd', 'nc', 'dx', 'disdx', 'ra'],
                           [plotter.plot_timeseries, plotter.plot_power_spectral_density, plotter.plot_noise_color,
                            plotter.plot_absolute_step, plotter.plot_width_distribution_ratios,
                            plotter.plot_rank_abundance]):
            if c in self.composition:
                func(self.axes[c][i], raw=True)

        if 'nn' in self.composition:
            plotter.plot_neutrality_measures(self.axes['KL'][i], self.axes['NCT'][i], raw=True)

    def figure(self):
        return self.fig


class PlotNoiseColorComparison():
    def __init__(self, files, labels, selfints=1, legend_title=None, ax=0, masi=True, interaction_colors=False):
        if ax == 0:
            self.fig = plt.figure(figsize=(4, 3.5), tight_layout=True)
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = ax

        self.ax.set_xscale('log')

        if masi == True:
            self.xaxis = 'masi'
        else:
            self.xaxis = 'ma'

        self.interaction_colors = interaction_colors

        if isinstance(selfints, float) or isinstance(selfints, int):
            self.selfints = [selfints] * len(files)
        elif len(selfints) < len(files):
            raise IndexError("The length of the self-interactions must be equal to the length of the files.")
        else:
            self.selfints = selfints

        self.legend_title = legend_title

        for file, label, selfint in zip(files, labels, self.selfints):
            self.plot_file(file, label, selfint)

        self.label_axes()

        # legend entries in opposite order:
        self.invert_legend_entries()

        self.plot_background_colors()

    def plot_file(self, file, label, selfint):
        if isinstance(file, str):
            df = pd.read_csv(file, index_col=0, na_values='NAN')
        elif isinstance(file, pd.DataFrame):
            df = file.copy()
        df.dropna(how='all', axis='index', inplace=True)

        if 'steady state' in df.columns:  # files created without interactions
            ss = df['steady state']

            df = df[[col for col in df if col.endswith('slope')]]

            if self.xaxis == 'masi':
                x = ss * selfint
            elif self.xaxis == 'ma':
                x = ss

            self.ax.errorbar(x, np.mean(df.T), np.std(df.T), linestyle='', marker='.', label=label)
        else:  # files created with interactions have different structure
            means = df.loc['means']
            stds = df.loc['stds']
            if "KL" in df.index:
                KL = df.loc['KL']
            mean_color = df.loc['mean_color']
            std_color = df.loc['std_color']

            if self.interaction_colors:
                c = self.interaction_mapper().to_rgba(float(label))
                self.ax.errorbar(means, mean_color, std_color, label=label, linestyle='', marker='.', c=c)
            else:
                self.ax.errorbar(means, mean_color, std_color, label=label, linestyle='', marker='.')

    def interaction_mapper(self):
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.21, clip=True)
        return cm.ScalarMappable(norm=norm, cmap='summer')

    def label_axes(self):
        if self.xaxis == 'masi':
            self.ax.set_xlabel(r'Mean abundance $\times$ self-interaction', ha='right', x=1)
        else:
            self.ax.set_xlabel(r'Mean abundance', ha='right', x=1)
        self.ax.set_ylabel('Slope power spectral density')

    def invert_legend_entries(self):
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles[::-1], labels[::-1], title=self.legend_title, loc=2)

    def change_number_columns_legend(self, ncol):
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, title=self.legend_title, loc=2, ncol=ncol)

    # TODO make dependent on ranges
    def plot_background_colors(self):
        x = np.linspace(0.9, -3, 500).reshape([500, 1])
        if self.ax.get_xscale() == 'log':
            self.background = self.ax.imshow(x, cmap=noise_cmap_ww, vmin=noise_lim[0], vmax=noise_lim[1],
                                             extent=(1e-3, 200, -3, 0.9), aspect='auto', alpha=0.75)
        else:
            self.background = self.ax.imshow(x, cmap=noise_cmap_ww, vmin=noise_lim[0], vmax=noise_lim[1],
                                             extent=(-5, 105, -3, 0.9), aspect='auto', alpha=0.75)

    def figure(self):
        return self.fig

    def set_limits(self, limits):
        left, right, bottom, top = limits

        left_orig, right_orig = self.ax.get_xlim()
        bottom_orig, top_orig = self.ax.get_ylim()

        if left < left_orig or right > right_orig or top > top_orig or bottom < bottom_orig:
            self.background.remove()

            x = np.linspace(0.9, -3, 500).reshape([500, 1])
            if self.ax.get_xscale() == 'log':
                self.background = self.ax.imshow(x, cmap=noise_cmap_ww, vmin=noise_lim[0], vmax=noise_lim[1],
                                                 extent=(left, right, bottom, top), aspect='auto', alpha=0.75)
            else:
                self.background = self.ax.imshow(x, cmap=noise_cmap_ww, vmin=noise_lim[0], vmax=noise_lim[1],
                                                 extent=(left, right, bottom, top), aspect='auto', alpha=0.75)

        self.ax.set_xlim([left, right])
        self.ax.set_ylim([bottom, top])


def main():
    print('test plotting')

    ts = sglv_timeseries.glv.Timeseries.main().timeseries

    ts2 = sglv_timeseries.glv.Timeseries.main().timeseries

    fig = PlotTimeseriesComparison([ts, ts2])

    plt.show()


if __name__ == "__main__":
    main()
