def check_consistency_noise_color(fts, ffig):
    transient = 100
    a = transient + 1000
    b = a + 1000

    species = [0,1]
    data = np.loadtxt(fts)

    fig = plt.figure()
    ax = fig.add_subplot(1,4,1)
    ax.plot(data[species, transient:b].T)
    for i in [a]:
        ax.axvline(x=i)

    axa = fig.add_subplot(1, 4, 2)
    axb = fig.add_subplot(1, 4, 3, sharex=axa, sharey=axa)
    axt = fig.add_subplot(1, 4, 4, sharex=axa, sharey=axa)

    for s in species:
        example_noise_fit(axa, data[s, transient:a])
        example_noise_fit(axb, data[s, a:b])
        example_noise_fit(axt, data[s, transient:b])

    axa.legend()
    axb.legend()
    axt.legend()

    if False:
        df = pd.DataFrame(data.T, columns=['species_%d' % i for i in range(100)])
        df2 = pd.DataFrame(data[:,100:].T, columns=['species_%d' % i for i in range(100)])

        print(noise_slope(df))
        print(noise_slope(df2))

    plt.show()
    #plt.savefig(ffig)

def check_consistency_noise_color2(fts):
    data = np.loadtxt(fts)

    halfway = int(len(data[0])/2)

    fig = plt.figure(tight_layout=True)

    df = pd.DataFrame(data[:,:halfway].T, columns=['species_%d' % i for i in range(100)])
    df2 = pd.DataFrame(data[:,halfway:].T, columns=['species_%d' % i for i in range(100)])

    n1 = noise_slope(df)
    n2 = noise_slope(df2)

    idx = np.where(abs(n1['slope_linear']-n2['slope_linear']) > 1)[0][0]

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.grid()

    example_noise_fit(ax, df['species_%d'%(idx)], label='%.2f'%n1['slope_linear'][idx])
    example_noise_fit(ax, df2['species_%d'%(idx)], label='%.2f'%n2['slope_linear'][idx])
    ax.legend()

    df = pd.DataFrame(data[:, 100:halfway].T, columns=['species_%d' % i for i in range(100)])
    #df2 = pd.DataFrame(data[:, halfway+100:].T, columns=['species_%d' % i for i in range(100)])

    n1_notrans = noise_slope(df)
    #n2_notrans = noise_slope(df2)

    ax = fig.add_subplot(2,1,1)
    ax.set_title('with transient in first half')

    ax.scatter(n1['slope_linear'], n2['slope_linear'], label=None)
    e = n1['slope_linear'] - n2['slope_linear']
    mse = np.mean(e[np.isfinite(e)] ** 2)
    ax.plot([-3,0], [-3,0], label='MSE = %.2f' % mse)
    ax.set_xlabel('slope noise first half')
    ax.set_ylabel('slope noise second half')
    ax.legend()

    ax = fig.add_subplot(2,1,2)
    ax.set_title('without transient in first half')

    ax.scatter(n1_notrans['slope_linear'], n2['slope_linear'], label=None)
    e = n1_notrans['slope_linear'] - n2['slope_linear']
    mse = np.mean(e[np.isfinite(e)]**2)
    ax.plot([-3,0], [-3,0], label='MSE = %.2f' % mse)
    ax.set_xlabel('slope noise first half')
    ax.set_ylabel('slope noise second half')
    ax.legend()


    #plt.show()
    #plt.savefig(ffig)
