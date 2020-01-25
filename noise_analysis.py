from GLV import *
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit

from generate_timeseries import *

def noise_color(data_input): # TODO previously noise_slope
    """ Returns panda dataframe with the slope of the noise of the species of which the timeseries is given in input file f"""

    if isinstance(data_input, str):
        if data_input.endswith('.csv'):
            data = pd.read_csv(data_input, index_col=0)
        elif data_input.endswith('.txt'):
            data = np.loadtxt(data_input)
            cols = ['species_%d' % i for i in range(1, len(data_input) + 1)]
            data = pd.DataFrame(data.T, columns=cols)
    elif isinstance(data_input, np.ndarray):
        cols = ['species_%d' % i for i in range(1, len(data_input) + 1)]
        data = pd.DataFrame(data_input.T, columns=cols)
    elif isinstance(data_input, pd.DataFrame):
        data = data_input.drop('time', axis='columns')
    elif not isinstance(data_input, pd.DataFrame):
        print("Please use dataframe or filename.")
        return

    results = pd.DataFrame(index=data.columns)

    results['slope_linear'] = np.nan
    results['std_slope_linear'] = np.nan
    results['pvalue'] = np.nan
    results['slope_spline'] = np.nan

    for c in [x for x in data.columns if x.startswith('species_')]:
        if sum(data[c] == 0) > 0.9 * len(data[c]):
            continue;

        # Fourier transform
        frq, f = signal.periodogram(data[c])

        frq = np.log10(frq[1:])  # cut zero -> log(0) = -INF
        f = np.log10(np.abs(f)[1:])

        frq = frq[~np.isnan(f)]
        f = f[~np.isnan(f)]

        frq = frq[np.isfinite(f)]
        f = f[np.isfinite(f)]

        if len(frq) > 5:
            p_spline = get_natural_cubic_spline_model(frq, f, minval=min(frq), maxval=max(frq), n_knots=3.5)

            y = p_spline.predict(frq)

            deriv = (y[1:] - y[:-1]) / (frq[1:] - frq[:-1])

            results.loc[c]['slope_spline'] = min(deriv)

            # only consider frequencies which correspond to periods that are smaller than (length_timeseries/10)
            # otherwise effects from windowing
            f = f[frq >= min(frq) + 1]
            frq = frq[frq >= min(frq) + 1]

            # linear fit
            p_lin, cov = np.polyfit(frq, f, deg=1, cov=True)

            results.loc[c]['slope_linear'] = p_lin[0]
            results.loc[c]['std_slope_linear'] = cov[0, 0]

            # confidence in model
            df = pd.DataFrame(columns=['frq', 'f'])
            df['frq'] = frq
            df['f'] = f

            lm1 = smf.ols(formula='frq ~ f', data=df).fit()

            results.loc[c]['pvalue'] = lm1.pvalues['f']
    return results

def compare_noise_profiles(f):
    figl = plt.figure("linear interpolation", figsize=(8, 8))
    figs = plt.figure("spline interpolation", figsize=(8, 8))

    falpha = '{:.3f}'
    fconnec = '{:.2f}'

    gs = gridspec.GridSpec(5, 5, hspace=0.02, wspace=0.02)

    for i, (c, ii) in enumerate(zip([0.8, 0.6, 0.4, 0.25, 0.1], [1, 7, 5, 2, 0])):
        for j, (alpha, jj) in enumerate(zip([0.02, 0.05, 0.1, 0.15, 0.2], [0, 2, 5, 7, 1])):

            # every connectivity-interaction_strength combination has a new graph
            # one linear (l) and one spline (s)
            axl = figl.add_subplot(gs[5 * i + j])
            axs = figs.add_subplot(gs[5 * i + j])

            # Fix axis such that differnt subgraphs can be compared
            axl.set_xlim([-5, 0])
            axl.set_ylim([0, 30])

            axs.set_xlim([-5, 0])
            axs.set_ylim([0, 30])

            # Set labels, remove ticks between figures (limits are all equal, see code right above)
            if i < 4:
                plt.setp(axl.get_xticklabels(), visible=False)
                plt.setp(axs.get_xticklabels(), visible=False)
            else:
                axl.set_xlabel(r'$\alpha$ = ' + falpha.format(alpha))
                axs.set_xlabel(r'$\alpha$ = ' + falpha.format(alpha))
            if j > 0:
                plt.setp(axl.get_yticklabels(), visible=False)
                plt.setp(axs.get_yticklabels(), visible=False)
            else:
                axl.set_ylabel(r'C = ' + fconnec.format(c))
                axs.set_ylabel(r'C = ' + fconnec.format(c))

            # plot histograms of different examples on top of each other
            for l in ['a', 'b', 'c', 'd', 'e']:
                fts = f + '%d%d%s.csv' % (ii, jj, l)

                # get noise profile

                noise = noise_color(fts)
                noise.dropna(axis='rows', inplace=True)
                noise = noise[noise.pvalue <= 0.05]

                # plot noise profiles in histograms

                axl.hist(noise['slope_linear'], alpha=0.2, bins=np.linspace(-5, 0, 50))
                axs.hist(noise['slope_spline'], alpha=0.2, bins=np.linspace(-5, 0, 50))

                if False:
                    print("total amount of solutions: %d, number original species: %d" % (
                    len(noise), len(data.columns) - 1))
                    print("linear, black: %.1f%%, brown: %.1f%%, pink: %.1f%%, white:%.1f%%" % (
                        sum(noise['slope_linear'] < -2.25) / len(noise['slope_linear']) * 100,
                        sum(np.logical_and(-2.25 < noise['slope_linear'], noise['slope_linear'] < -1.75)) / len(noise) * 100,
                        sum(np.logical_and(-1.75 < noise['slope_linear'], noise['slope_linear'] < -0.5)) / len(noise) * 100,
                        sum(-0.5 < noise['slope_linear']) / len(noise) * 100))

                    print("spline, black: %.1f%%, brown: %.1f%%, pink: %.1f%%, white:%.1f%%" % (
                        sum(noise['slope_spline'] < -2.25) / len(noise) * 100,
                        sum(np.logical_and(-2.25 < noise['slope_spline'], noise['slope_spline'] < -1.75)) / len(noise) * 100,
                        sum(np.logical_and(-1.75 < noise['slope_spline'], noise['slope_spline'] < -0.5)) / len(noise) * 100,
                        sum(-0.5 < noise['slope_spline']) / len(noise) * 100))

            # plot vertical lines at limitations of black, brown, pink and white as in Faust 2018 Microbiome
            for bin_edge in [-2.25, -1.75, -0.5]:
                axl.axvline(x=bin_edge, linestyle=':', linewidth=0.5)
                axs.axvline(x=bin_edge, linestyle=':', linewidth=0.5)

    # save figures

    #plt.figure("linear interpolation")
    #plt.savefig(f + "linear.png")

    #plt.figure("spline interpolation")
    #plt.savefig(f + "spline.png")
    return

def compare_noise_implementations():
    f_linear = 'results/data_linear_77a.csv'
    f_sqrt = 'results/data_sqrt_77a.csv'
    f_constant = 'results/data_constant_77a.csv'
    f_small = 'results/data_sqrt_small_77a.csv'
    f_large = 'results/data_sqrt_large_77a.csv'
    f_largets = 'results/data_sqrt_largetimestep_77a.csv'
    f_perturb = 'results/data_sqrt_perturbation_77a.csv'

    fs = [f_linear, f_sqrt, f_constant, f_small, f_large, f_largets, f_perturb]
    titles = ['linear', 'sqrt', 'constant', 'small', 'large', 'largetimestep', 'perturbation']

    tss = [pd.read_csv(f, index_col=0) for f in fs]

    noise = [noise_color(ts) for ts in tss]

    N = 3 # number of species to plot
    #spec_to_plot = ts_linear.columns[ts_linear.max() > 0][:N]
    spec_to_plot = ['species_1', 'species_2']

    figts = plt.figure("timeseries")
    gsts = gridspec.GridSpec(3,len(tss), height_ratios=[1.5,1,1])

    fign = plt.figure("noise")
    gsn = gridspec.GridSpec(1,len(tss))

    for i in range(len(tss)):
        # plot complete timeseries
        if i == 0:
            ax = figts.add_subplot(gsts[i])
        else:
            ax = figts.add_subplot(gsts[i], sharey=ax)
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_title(titles[i])
        ax.plot(tss[i][spec_to_plot])

        # plot zoom of timeseries
        if i == 0:
            ax_zoom1 = figts.add_subplot(gsts[len(tss) + i])
        else:
            ax_zoom1 = figts.add_subplot(gsts[len(tss) + i], sharey=ax_zoom1)
            plt.setp(ax_zoom1.get_yticklabels(), visible=False)
        ax_zoom1.plot(tss[i][spec_to_plot[0]].iloc[:500])

        if i == 0:
            ax_zoom2 = figts.add_subplot(gsts[2*len(tss) + i])
        else:
            ax_zoom2 = figts.add_subplot(gsts[2*len(tss) + i], sharey=ax_zoom2)
            plt.setp(ax_zoom2.get_yticklabels(), visible=False)
        ax_zoom2.plot(tss[i][spec_to_plot[1]].iloc[:500], color='orange')
        if i > 0:
            ax_zoom2.set_ylabel('')

        # plot noise fit
        if i == 0:
            axn = fign.add_subplot(gsn[i])
        else:
            axn = fign.add_subplot(gsn[i], sharey=axn)
            plt.setp(axn.get_yticklabels(), visible=False)
        example_noise_fit(axn, tss[i][spec_to_plot[1]].iloc[:500])

    fig = plt.figure("compare slopes")
    gs = gridspec.GridSpec(2,round((len(tss)-1)/2 + 0.1)) # + 0.1 to account for bankers rounding
    print(len(tss), (len(tss)-1)/2, round((len(tss)-1)/2))

    for i in range(len(tss)-1):
        if i == 0:
            ax = fig.add_subplot(gs[i])
            ax.set_ylabel(titles[0])
        else:
            print(i)
            ax = fig.add_subplot(gs[i], sharey=ax, sharex=ax)
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.scatter(noise[i+1]['slope_linear'], noise[0]['slope_linear'])
        ax.set_xlabel(titles[i+1])
        ax.plot([-5,0], [-5,0]) #diagonal

def reproduce_timeseries_Faust():
    fts = 'reproduce_translate_Faust/Faust_repro3b.csv'
    #fomega = np.loadtxt('../enterotypes_ibm/Rickermatrices-Faust/9_interactionmatrix.txt')
    fomega = np.loadtxt('reproduce_translate_Faust/omega3.txt')

    #forig = '../enterotypes_ibm/timeseries-Faust/1_timeseries/1_timeseries.txt'
    #m = np.mean(np.loadtxt(forig), axis=1, keepdims=True)
    #fg = -fomega.dot(m)

    m = np.loadtxt('reproduce_translate_Faust/K3.txt')
    fg = -fomega.dot(m)

    fmu = None
    SIS = False

    generate_timeseries_noise([0, 0, 0], fts, fomega, fg, fmu, SIS, noise = 0.05, noise_implementation=NOISE.LANGEVIN_LINEAR)

def check_code_on_Faust_timeseries(f_ricker, ffig):
    save = True
    fit = 'slope_linear'

    noise = noise_color(f_ricker)

    fig = plt.figure(figsize=(10,6), tight_layout=True)
    gs = gridspec.GridSpec(3,3)

    ax = fig.add_subplot(gs[0])
    ax.set_title('noise profile')
    ax.hist(noise[fit][np.isfinite(noise[fit])], alpha=0.2, bins=np.linspace(-5, 0, 50))

    # plot vertical lines at limitations of black, brown, pink and white as in Faust 2018 Microbiome
    for bin_edge in [-2.25, -1.75, -0.5]:
        ax.axvline(x=bin_edge, linestyle=':', linewidth=0.5)

    # plot one timetrace for every noise color

    if f_ricker.endswith('.txt'):
        data = np.loadtxt(f_ricker)
    elif f_ricker.endswith('.csv'):
        data = pd.read_csv(f_ricker, index_col=0).values.T

    noise_edges = [-np.inf, -2.25, -1.75, -0.5, np.inf]
    noise_color = ['black', 'brown', 'pink', 'white']
    gss = [gs[3], gs[4], gs[6], gs[7]]

    for i in range(len(noise_color)):
        ax = fig.add_subplot(gss[i])
        ax.set_title(noise_color[i])

        # all timetraces of specific noise_color
        tt = np.logical_and(noise_edges[i] < noise[fit].values, noise[fit].values < noise_edges[i+1])
        if sum(tt) > 0:
            idx = next(ii for ii, v in enumerate(tt) if v)
            ax.plot(data[idx][:1000])

    ax = fig.add_subplot(gs[1])
    ax.set_title('relative noise fluctuations')
    y = ((data[:,1:] - data[:,:-1])/data[:,1:]).flatten()
    y = y[np.isfinite(y)]

    ax.hist(y, density=True, bins=np.linspace(-0.5,0.5, 50))
    x = np.linspace(-0.5,0.5,500)
    mu, std = norm.fit(y)
    ax.plot(x, norm.pdf(x, mu, std), label='$\mu =$ %.2f, \n $\sigma = $ %.3f' % (mu, std))
    ax.plot(x, norm.pdf(x, 0, 0.05))
    ax.legend()

    # plot total number of individuals in population, see if constant

    ax = fig.add_subplot(gs[2])
    ax.set_title('total number population')
    y = np.sum(data, axis=0)
    ax.plot(y)
    ax.set_ylim([0,1.1*max(y)])

    # plot mean abundances of species

    ax = fig.add_subplot(gs[1:, -1])
    y = np.mean(data, axis=1)
    ax.scatter(y, noise[fit], cmap=noise_cmap, vmin=noise_lim[0], vmax=noise_lim[1], c=noise[fit], label=None)
    p_lin = np.polyfit(y, noise[fit], deg=1)
    x = np.linspace(min(y), max(y))
    ax.plot(x, p_lin[0]*x + p_lin[1], label='y = %.2fx + %.2f' % (p_lin[0], p_lin[1]))
    ax.set_xlabel('mean abundance')
    ax.set_ylabel('noise color')
    ax.legend()

    if save:
        plt.savefig(ffig)

def compare_noise_abundance_dependencies(fts = [], ffig = 0, labels = []):
    save = True
    fit = 'slope_linear'

    fig = plt.figure(figsize=(8,4), tight_layout=True)
    ax = fig.add_subplot(121)
    axb = fig.add_subplot(122, sharey=ax)

    bounds = [0, 0.5, 1, 1.5]  # ,2]

    fig2 = plt.figure(figsize=(8,8),tight_layout=True)
    gs = gridspec.GridSpec(len(fts),len(bounds)-1)

    fig3 = plt.figure(figsize=(8,8),tight_layout=True)

    x = np.linspace(2, -3, 500).reshape([500,1])
    ax.imshow(x, cmap=noise_cmap_ww, vmin = noise_lim[0], vmax= noise_lim[1], extent=(0, 2, -3, 2), aspect='auto', alpha=0.75)
    ax.set_xlim([0, 2])
    axb.imshow(x, cmap=noise_cmap_ww, vmin = noise_lim[0], vmax= noise_lim[1], extent=(0, 1, -3, 2), aspect='auto', alpha=0.75)
    #axb.set_ylim([-3,2])

    maxx = 0

    for i, f in enumerate(fts):
        noise = noise_color(f)

        if f.endswith('.txt'):
            data = np.loadtxt(f)
        elif f.endswith('.csv'):
            data = pd.read_csv(f, index_col=0).values.T

        y = np.mean(data, axis=1)
        #y /= np.sum(y)
        if len(labels) == 0:
            label = None
        else:
            label = labels[i]
        n = noise[fit]
        ebn = noise['std_slope_linear']

        data = data[np.isfinite(n)]
        ebn = ebn[np.isfinite(n)]
        y = y[np.isfinite(n)]
        n = n[np.isfinite(n)]

        s = ax.scatter(y, n, s=5, label=label)
        ax.errorbar(y, n, ebn, linestyle='None', label=None)
        axb.scatter(y/sum(y), n, s=5, label=label)
        axb.errorbar(y/sum(y), n, ebn, linestyle='None', label=None)

        #p_lin = np.polyfit(y, n, deg=2) # without errorbars

        # with errorbars
        def mypoly(x, a, b, c):
            return np.polyval([a,b,c], x)

        p_lin, _ = curve_fit(mypoly, y, n, p0=[1,1,1], sigma=ebn)

        print("p_lin", p_lin)

        x = np.linspace(min(y), max(y))
        l = ax.plot(x, p_lin[0]*x**2 + p_lin[1]*x + p_lin[2], color=s.get_facecolors()[0]) #, label='y = %.2fx + %.2f' % (p_lin[0], p_lin[1]))

        #p_lin = np.polyfit(y/sum(y), n, deg=2)

        p_lin, _ = curve_fit(mypoly, y/sum(y), n, p0=[1, 1, 1], sigma=ebn)

        print("p_lin2", p_lin)

        x = np.linspace(0, 1.1*max(y/sum(y)))
        maxx = max(maxx, 1.1*max(y/sum(y)))
        l = axb.plot(x, p_lin[0] * x ** 2 + p_lin[1] * x + p_lin[2],
                    color=s.get_facecolors()[0])  # , label='y = %.2fx + %.2f' % (p_lin[0], p_lin[1]))

        for j in range(len(bounds) - 1):
            lb = bounds[j]
            rb = bounds[j + 1]

            idces = np.where((lb < y) & (y < rb))[0] #next((i for i,x in enumerate(y) if lb < x < rb), None)

            if i == 0:
                ax2 = fig2.add_subplot(gs[i, j])
                ax2.set_title('%.1f < x < %.1f' % (lb, rb))

                ax3 = fig3.add_subplot(gs[i, j])
                ax3.set_title('%.1f < x < %.1f' % (lb, rb))

                if len(idces) == 0:
                    ax2.axes.get_yaxis().set_visible(False)
                    ax2.axes.get_xaxis().set_visible(False)
                    ax3.axes.get_yaxis().set_visible(False)
                    ax3.axes.get_xaxis().set_visible(False)
                    ax2.set_frame_on(False)
                    ax3.set_frame_on(False)

            if len(idces) > 0:
                y2 = (data[idces,1:] - data[idces,:-1])/data[idces,:-1]

                y2 = y2[np.isfinite(y2)]

                ax2 = fig2.add_subplot(gs[i, j])
                ax2.grid()

                ax2.hist(y2, bins = np.linspace(-1.1,1.5,100), color=l[0].get_color())
                at = AnchoredText("%.2f %% < 0" % (sum(y2 < 0)/len(y2)*100), loc=2)
                ax2.add_artist(at)

                ax3 = fig3.add_subplot(gs[i, j])
                ax3.grid()

                if i == 0:
                    ax3.set_title('%.1f < x < %.1f' % (lb, rb))

                ax3.hist(data[idces].flatten(), bins=np.linspace(0, 1.5, 100), color=l[0].get_color())
                #at = AnchoredText("%.2f %% < " % (sum(data[idces].flatten() < 1) / len(data[idces].flatten()) * 100), loc=2)
                #ax3.add_artist(at)

    ax.set_xlabel('mean abundance')
    ax.set_ylabel('noise color')
    ax.legend(title='$\sigma$')
    axb.set_xlabel('relative mean abundance')
    axb.set_ylabel('noise color')
    axb.set_xlim([0, maxx])

    if save and len(ffig) > 0:
        fig.savefig(ffig[0])
        fig2.savefig(ffig[1])
        fig3.savefig(ffig[2])

def information_ts(f):
    plottimeseries(pd.read_csv(f, index_col=0, na_values='NAN'))
    plt.show()

    y_ts = pd.read_csv(f, index_col=0).values
    s = len(y_ts[0])
    mean = np.mean(y_ts, axis=0)
    mean_N = np.mean(mean)

    # x bar = J/S

    cov = np.zeros([S,S])

    for i in range(S):
        for j in range(S):
            cov[i,j] = np.mean(
                (y_ts[:,i] - mean[i])*(y_ts[:,j] - mean[j]))

    cov_diag0 = np.copy(cov); np.fill_diagonal(cov_diag0, 0)
    cov_N = np.sum(cov_diag0)/S/(S-1)*np.ones([S,S])
    np.fill_diagonal(cov_N, np.mean(np.diag(cov)))

    KL = 1/2 * (np.trace(np.dot(np.linalg.inv(cov_N), cov)) +
            (((mean_N - mean).T).dot(np.linalg.inv(cov_N))).dot(mean_N - mean)
           - np.linalg.matrix_rank(cov) + np.log(np.linalg.det(cov_N)/np.linalg.det(cov)))

    mat = plt.matshow(cov)
    plt.colorbar(mat)
    plt.show()

    print("trace", np.trace(np.dot(np.linalg.inv(cov_N), cov)))
    print("rank", -np.linalg.matrix_rank(cov))
    print("mean", (((mean_N - mean).T).dot(np.linalg.inv(cov_N))).dot(mean_N - mean))
    print("determinant", np.log(np.linalg.det(cov_N)/np.linalg.det(cov)))
    print("det neutral, det cov", np.linalg.det(cov_N), np.linalg.det(cov))
    print("Kullback Leibler", KL)

# compare linear and spline fit
if False:
    fig = plt.figure(figsize=(3.5, 3), tight_layout=True)
    ax = fig.add_subplot(111)
    diff = results['slope_spline'] - results['slope_linear']
    ax.hist(diff, bins=np.linspace(-2, 1, 50))
    ax.set_xlabel('Difference spline slope and linear slope')
    results['slopediff'] = results['slope_spline'] - results['slope_linear']
    print(results)
    plt.savefig('glv/noise/difference_spline_linear.png')

    # plt.savefig('glv/noise/lineartest_color_mig.png')
    # plt.show()

# plot timeseries and zoom
if False:
    f = 'glv/noise/data_test_00a.csv'
    data = pd.read_csv(f)
    time = data['timestep']

    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(2, 3, (1, 4))
    ax.plot(range(len(data['species_1'])), data['species_1'])
    ax.plot(range(len(data['species_2'])), data['species_2'])
    ax.set_ylim([0, 1.1 * data['species_1'].max()])

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(range(len(data['species_1'])), data['species_1'])
    ax.set_ylim([0.95 * data['species_1'].min(), 1.05 * data['species_1'].max()])
    plt.setp(ax.get_yticklabels(), visible=False)
    # ax.yaxis.tick_right()

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(range(100), data['species_1'][:100])
    ax.set_ylim([0.95 * data['species_1'].min(), 1.05 * data['species_1'].max()])
    ax.yaxis.tick_right()

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(range(len(data['species_2'])), data['species_2'], c='orange')
    ax.set_ylim([0.95 * data['species_2'].min(), 1.05 * data['species_2'].max()])
    plt.setp(ax.get_yticklabels(), visible=False)
    # ax.yaxis.tick_right()

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(range(100), data['species_2'][:100], c='orange')
    ax.set_ylim([0.95 * data['species_2'].min(), 1.05 * data['species_2'].max()])
    ax.yaxis.tick_right()
    plt.savefig('glv/noise/timeseries.png')
    plt.show()

def figure_overview_noise_params(figfiles, f, f_omega, f_g):
    data = pd.read_csv(f)
    spec_col = [x for x in data.columns if x.startswith('species_')]
    data = data[spec_col]
    omega = pd.read_csv(f_omega, index_col=0)
    g = pd.read_csv(f_g, index_col=0)

    # save = False

    save = True

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(4, 2, width_ratios=[5, 3], height_ratios=[1] * 4)
    gscb = gridspec.GridSpec(16, 2, width_ratios=[5, 3], height_ratios=[1] * 4 * 4)
    cb = [gscb[1, 1], gscb[5, 1], gscb[9, 1], gscb[13, 1]]

    ax = fig.add_subplot(gs[0, 0])
    ms = ax.matshow(omega.T[:5], cmap='coolwarm', vmin=-np.max(abs(omega.values)), vmax=np.max(abs(omega.values)),
                    extent=[0, 100, 0, 5], aspect=3)
    ax.set_yticks(range(5))
    ax.set_yticklabels(range(1, 6), fontsize=7)
    ax.set_xticks(range(4, len(omega), 5))
    ax.set_xticklabels(np.arange(5, len(omega) + 1, 5), fontsize=7)

    ax = fig.add_subplot(cb[0])
    ax.set_title('Interaction matrix', y=1.5)
    plt.colorbar(ms, cax=ax, orientation='horizontal', aspect=50)

    ax = fig.add_subplot(gs[1, 0])
    ms = ax.matshow(g.T, cmap='coolwarm', extent=[0, 100, 0, 1], aspect=5)
    ax.set_yticks([])
    ax.set_xticks(range(4, len(omega), 5))
    ax.set_xticklabels(np.arange(5, len(omega) + 1, 5), fontsize=7)

    ax = fig.add_subplot(cb[1])
    ax.set_title('Growthrates', y=1.2)
    plt.colorbar(ms, cax=ax, orientation='horizontal', aspect=2)

    data[data < 1e-30] = 0

    ax = fig.add_subplot(gs[2, 0])
    ms = ax.matshow(np.log10(data.mean().values).reshape((g.T).shape), cmap='coolwarm', extent=[0, 100, 0, 1], aspect=5)
    ax.set_yticks([])
    ax.set_xticks(range(4, len(omega), 5))
    ax.set_xticklabels(np.arange(5, len(omega) + 1, 5), fontsize=7)

    ax = fig.add_subplot(cb[2])
    ax.set_title('Mean abundance (log)', y=1.2)
    plt.colorbar(ms, cax=ax, orientation='horizontal', aspect=2)

    slope = noise_color(f)['slope_linear']

    ax = fig.add_subplot(gs[3, 0])
    ms = ax.matshow(slope, cmap=noise_cmap, vmax=noise_lim[1], vmin=noise_lim[0], extent=[0, 100, 0, 1], aspect=5)
    ax.set_yticks([])
    ax.set_xticks(range(4, len(omega), 5))
    ax.set_xticklabels(np.arange(5, len(omega) + 1, 5), fontsize=7)

    ax = fig.add_subplot(cb[3])
    ax.set_title('Noise color/slope', y=1.2)
    plt.colorbar(ms, cax=ax, orientation='horizontal', aspect=2)

    plt.subplots_adjust(left=0.1, bottom=0.08, right=0.92, top=0.95, wspace=0.1, hspace=0.1)

    if save:
        plt.savefig(figfiles[0])

    # figure with correlations

    plt.rcParams["axes.grid"] = True

    fig = plt.figure(figsize=(9, 3), tight_layout=True)

    gs = gridspec.GridSpec(1, 3, wspace=0)

    ax = fig.add_subplot(gs[0])
    ax.scatter(omega.iloc[:, 0], slope, cmap=noise_cmap, c=slope, vmin=noise_lim[0], vmax=noise_lim[1])
    ax.set_xlabel('Influence of SIS')
    ax.set_ylabel('Noise slope')

    ax = fig.add_subplot(gs[1], sharey=ax)
    ax.scatter(g, slope, cmap=noise_cmap, c=slope, vmin=noise_lim[0], vmax=noise_lim[1])
    ax.set_xlabel('Growthrates')
    plt.setp(ax.get_yticklabels(), visible=False)

    ax = fig.add_subplot(gs[2], sharey=ax)
    ax.scatter(np.log10(data.mean().values), slope, cmap=noise_cmap, c=slope, vmin=noise_lim[0], vmax=noise_lim[1])
    ax.set_xlabel('log$_{10}$(Mean abundance)')
    plt.setp(ax.get_yticklabels(), visible=False)

    if save:
        plt.savefig(figfiles[1])

        # plt.show()

# overview_loop_params

if False:
    for i in [0, 2, 5, 7, 1]:
        for j in [0, 2, 5, 7, 1]:
            figfile = ['glv/noise/analysis_%d%df.png' % (i, j), 'glv/noise/analysis2_%d%df.png' % (i, j)]
            f = 'glv/noise/data_SIS_wide_%d%df.csv' % (i, j)
            f_omega = 'glv/noise/omega_%d%df.csv' % (i, j)
            f_g = 'glv/noise/g_%d%df.csv' % (i, j)

            if os.path.exists(f):
                figure_overview_noise_params(figfile, f, f_omega, f_g)
            else:
                print('path not found')

if __name__ == "bla": #""__main__":
    #generate_timeseries_noise_loop_parameters()
    #compare_noise_profiles('results2/data_SIS_')
    #compare_noise_implementations()

    #plot_timeseries('self_ricker_nonoise.txt')

    #reproduce_timeseries_Faust()

    N = 100
    folder = 'reproduce_translate_Faust/'

    #fomega = np.random.uniform(0, 0.1, [N, N])
    #fomega *= np.random.choice([0, 1], [N, N], p=[0.8, 0.2])

    #fomega = np.zeros([N, N])
    #np.fill_diagonal(fomega, -1)

    #np.savetxt(folder + 'omega_Langevin_noint.txt', fomega)

    #K = np.random.uniform(0, 2, N)
    # K = np.ones(N)
    #fg = -fomega.dot(K)

    #np.savetxt(folder + 'g_Langevin_noint.txt', fg)

    # generate timeseries
    if False:
        N = 100
        noise = [0.5, 0.7, 0.9] #0.01,0.1,0.2,0.25,0.3]
        for ii, i in enumerate([]):# [6,7,8]):
            folder = 'reproduce_translate_Faust/'

            fts = folder + 'Arato_noint_%d.csv' % i

            #fomega = np.random.uniform(0, 0.1, [N, N])
            #fomega *= np.random.choice([0, 1], [N, N], p=[0.8, 0.2])

            #fomega = np.zeros([N, N])
            #np.fill_diagonal(fomega, -1)

            fomega = folder + 'omega_Langevin_noint.txt'

            #K = np.random.uniform(0,2,N)
            #K = np.ones(N)
            #fg = -fomega.dot(K)

            fg = folder + 'g_Langevin_noint.txt'

            fmu = None
            SIS = False

            generate_timeseries_noise([0, 0, 0], fts, fomega, fg, fmu, SIS, noise=noise[ii], noise_implementation=NOISE.ARATO_LINEAR)

        for i in [1]: #range(1, 16): #51):
            folder = '../enterotypes_ibm/timeseries-Faust/'

            f_ricker = folder + '%d_timeseries/%d_timeseries.txt' % (i, i)

            print(f_ricker)

            check_consistency_noise_color2(f_ricker)

            plt.savefig(folder + 'Faust_importance_transient_%d.png'  % i)

        for i in []: #[20, 21]: #range(2, 17):
            #folder = '../enterotypes_ibm/timeseries-Faust/'

            #f_ricker = folder + '%d_timeseries/%d_timeseries_notransient.txt' % (i, i)

            folder = 'reproduce_translate_Faust/'
            f_ricker = folder + 'self_ricker%d.txt' % i

            ffig = folder + 'self_ricker%d.png' % i

            check_code_on_Faust_timeseries(f_ricker, ffig)

    # folder = 'reproduce_translate_Faust/'
    # fts = [folder + 'Arato_noint_%d.csv' % i for i in [1,6,7,8]] #[1,2,3,4,5]]
    # labels = [0.01, 0.5, 0.7, 0.9] #[0.01, 0.1,0.2,0.25, 0.3]
    # ffigs = [folder + 'noise_abundance_noint_Aratob.png', folder + 'fluctuation_distribution_noint_Aratob.png', folder + 'ts_distribution_noint_Aratob.png']
    # compare_noise_abundance_dependencies(fts, ffigs, labels)
    #
    # plt.show()
    #
    # plt.figure()
    # plt.title('Arato, no interaction')
    # plot_timeseries('reproduce_translate_Faust/Arato_noint_3.csv')
    #
    # plt.figure()
    # plt.title('Langevin, no interaction')
    # plot_timeseries('reproduce_translate_Faust/Langevin_noint_3.csv')

    #plt.savefig('reproduce_translate_Faust/compare_ricker_Langevin_noint_largernoise.png')
    plt.show()

