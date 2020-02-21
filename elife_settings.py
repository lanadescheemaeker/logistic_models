import matplotlib as mpl
from cycler import cycler
import seaborn

def cm2inch(value):
    return value/2.54;

class ELIFE():
    TEXTWIDTH = cm2inch(14)  # 5.5
    FULLWIDTH = cm2inch(18.5)
    FONTSIZE = 7

    # a colorblind-friendly color cycle
    #COLOR_CYCLE = cycler(color = [(230,159,0), (86,180,233), (0,158,115), (0,114,178),
    #                              (213,94,0), (0,0,0), (204,121,167), (240,228,66)])
    COLOR_CYCLE = cycler(color=seaborn.color_palette('colorblind'))

def set_elife_settings():
    font = {'family': 'Open Sans', 'size': ELIFE.FONTSIZE}
    mpl.rc('font', **font)
    mpl.rc('legend', handlelength=1)

    mpl.rcParams['legend.fontsize'] = ELIFE.FONTSIZE
    mpl.rcParams['figure.titlesize'] = ELIFE.FONTSIZE
    mpl.rcParams['axes.titlesize'] = ELIFE.FONTSIZE
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['axes.labelpad'] = 4

    mpl.rcParams["legend.borderaxespad"] = 0.2
    mpl.rcParams["legend.handlelength"] = 0.8
    mpl.rcParams['lines.markersize'] = 3
    mpl.rcParams['lines.linewidth'] = 0.8

    mpl.rc('axes', prop_cycle=ELIFE.COLOR_CYCLE)