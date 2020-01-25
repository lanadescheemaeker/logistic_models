# code from https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

# code to make color lighter (0<amount<1) or darker (amount>1)
def change_color(color, amount=0.5):
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


# example code

if __name__ == "__main__":
    c = mcolors.ColorConverter().to_rgb
    rvb = make_colormap([c('red'), c('violet'), 0.33, c('violet'), c('blue'), 0.66, c('blue')])
    N = 1000
    array_dg = np.random.uniform(0, 10, size=(N, 2))
    colors = np.random.uniform(-2, 2, size=(N,))
    plt.scatter(array_dg[:, 0], array_dg[:, 1], c=colors, cmap=rvb)
    plt.colorbar()
    plt.show()