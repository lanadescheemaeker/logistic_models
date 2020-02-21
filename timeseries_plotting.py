import matplotlib.pyplot as plt

class PlotTimeseries():
    def __init__(self, ts, ax=None, species=None, raw=False):
        if ax == None:
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111)
        else:
            self.ax = ax

        self.ts = ts

        if species == None:
            self.selection = self.select_species()
        else:
            self.selection = species

        self.plot_timeseries(raw)
        self.ax.set_yscale('log')

    def select_species(self):
        self.mean = self.ts.mean()
        self.mean.drop('time', inplace=True)

        self.Nspecies = len(self.ts.columns) - 1

        sorted_species = self.mean.sort_values().index.tolist()[::-1]

        return sorted_species[::max(1, int(self.Nspecies / 4))]

    def plot_timeseries(self, raw):
        skip = max(1, int(len(self.ts) / 500))

        for s in self.selection:
            self.ax.plot(self.ts['time'][::skip], self.ts[s][::skip])

        if not raw:
            self.ax.set_ylabel('Abundance')
