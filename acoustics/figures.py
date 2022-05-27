##Author: 
##Date Started:
##Notes:

from pithy3 import *

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import pandas as pd

import resonance.resonance.limit_memory
import GT_color_palette as colors

# FIG PARAMS
rcParams['font.sans-serif'] = "Helvetica"
rcParams['font.family'] = "sans-serif"
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x  and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

class QuadrantFig:
    def __init__(self):
        clf()
        self.no_subplots = 3
        self.fig, self.axs = plt.subplots(
            self.no_subplots,
            2,
            figsize=(12, 10)
        )
        self.cap = (0, 1)
        self.amp = (0, 0)
        self.tof = (1, 0)
        self.pot = (2, 0)

    def plot_amplitude(self, dt, amplitudes):
        x = create_timedelta(dt)
        self.axs[self.amp].plot(x, amplitudes, c=colors.cu_colors[-1])

        self.axs[self.amp].set_xlabel('t [h]')
        self.axs[self.amp].set_ylabel('Amplitude ($\mathregular{|A|^{2}/|A_{0}|^{2}}$)')

    def plot_capacity(self, cycle_no, capacity):
        x = cycle_no + 1  # 0-based to 1-based
        self.axs[self.cap].scatter(
            x=x,
            y=capacity,
            color=colors.cu_colors[0],
            edgecolor='k',
            s=50
        )

        self.axs[self.cap].set_xlabel('Cycle no')
        self.axs[self.cap].set_ylabel('Capacity [mAh]')
        self.axs[self.cap].xaxis.set_major_locator(MaxNLocator(5))
        self.axs[self.cap].yaxis.set_label_position("right")
        self.axs[self.cap].yaxis.tick_right()

    def plot_neware(self, dt, I, V):
        x = create_timedelta(dt)
        self.axs[self.pot].plot(x, V, c='k')
        self.axs[self.pot].set_xlabel('t [h]')
        self.axs[self.pot].set_zorder(1)
        self.axs[self.pot].set_frame_on(False)

        ax2 = self.axs[self.pot].twinx()
        ax2.plot(x, I*1000, c='gray')

        ax2.set_ylabel('Current [mA]', c='gray')
        self.axs[self.pot].set_ylabel('Voltage [V]')

    def plot_tof_shift(self, dt, tof_shift):
        x = create_timedelta(dt)
        self.xlims = [x.iloc[0][0], x.iloc[-1][0]]
        self.axs[self.tof].plot(x, tof_shift*1000, c=colors.cu_colors[-1])

        self.axs[self.tof].set_xlabel('t [h]')
        self.axs[self.tof].set_ylabel('ToF shift [$\mu s$]')

    def set_properties(self, params):
        annotation = f"{params['cell_id']}\n{params['electrolyte']}\n{params['protocol']}"
        plt.text(
            0.001,
            0.80,
            annotation,
            fontweight="bold",
            fontsize=16,
            transform=self.axs[1,1].transAxes
        )

        # Set axes limits
        for i in range(0, self.no_subplots):
            self.axs[i, 0].set_xlim(self.xlims)
            if i < self.no_subplots-1:  # Remove xticks
                self.axs[i, 0].get_xaxis().set_visible(False)
        
        # self.fig.suptitle(params['cell_id'], fontsize=18)
        
        for i in range(1, 3):
            self.fig.delaxes(self.axs[i][1])

        plt.subplots_adjust(wspace=0.05, hspace=0.05, top=0.94)

##############################################

def create_timedelta(dt):
    timedelta = dt - dt.iloc[0]
    timedelta = timedelta / pd.Timedelta('1 hour')

    return timedelta

def plot_amplitude(dt, amplitudes):
    clf()
    fig, ax = plt.subplots()
    x = create_timedelta(dt)
    ax.plot(x, amplitudes, c=colors.cu_colors[-1])
    ax.set_xlabel('t [h]')
    ax.set_ylabel('Amplitude ($\mathregular{|A|^{2}/|A_{0}|^{2}}$)')

    showme()

def plot_capacity(cycle_no, capacity):
    clf()
    fix, ax = plt.subplots()
    x = cycle_no + 1  # 0-based to 1-based
    ax.scatter(
        x=x,
        y=capacity,
        color=colors.cu_colors[0],
        edgecolor='k',
        s=50
    )
    ax.set_xlabel('Cycle no')
    ax.set_ylabel('Capacity [mAh]')
    ax.xaxis.set_major_locator(MaxNLocator(5)) 

    showme()

def plot_modulus(dts, moduli, names, electrolytes):
    fig, ax = plt.subplots()    
    # colormap = GT_color_palette.generate_sequential_cu_colormap(N=len(moduli))
    colormap = plt.cm.coolwarm(np.linspace(0, 1, len(moduli)))
    for i, modulus in enumerate(moduli):
        # 1E-9 for axis in GPa
        ax.plot(
            dts[i],
            modulus * 1E-9,
            c=colormap[i],
            linewidth=1,
            label=f'{names[i][5:]}: {electrolytes[i]}'
        )

    # Properties    
    ax.set_title("Effective Young\'s Modulus", fontsize=16)
    ax.set_ylabel('E [GPa]')
    ax.set_xlabel('t [h]')
    ax.set_xlim([0, 80])
    ax.set_ylim([0, 10])
    ax.legend(title='cell')

def plot_neware(dt, I, V):
    clf()
    fig, ax = plt.subplots()
    x = create_timedelta(dt)

    ax.plot(x, V, c='k')
    ax.set_ylabel('Voltage [V]')
    ax.set_xlabel('t [h]')
    ax.set_zorder(1)
    ax.set_frame_on(False)
    ax2 = ax.twinx()
    ax2.plot(x, I*1000, c='gray')
    ax2.set_ylabel('Current [mA]', c='gray')

    showme()

def plot_tof_shift(dt, tof_shift):
    clf()
    fig, ax = plt.subplots()
    x = create_timedelta(dt)
    ax.plot(x, tof_shift*1000, c=colors.cu_colors[-1])

    ax.set_xlabel('t [h]')
    ax.set_ylabel('ToF shift [$\mu s$]')

    showme()


def plot_trigger(acoustic_amplitudes, trigger_index):
    """Plot for visually verifying that the trigger is at the correct index."""

    clf()
    plt.plot(acoustic_amplitudes, c=GT_color_palette.cu_colors[-1])
    plt.vlines(
        trigger_index,
        min(acoustic_amplitudes),
        max(acoustic_amplitudes),
        linestyle='dashed',
        color='gray',
        label='trigger'
    )
    plt.title('Getting ToF')
    plt.ylabel('Waveform amplitude [a.u.]')
    plt.xlabel('Samples [ ]')
    plt.legend()
    showme()

def plot_tofs(tofs):
    clf()
    plt.plot(tofs, c=GT_color_palette.cu_colors[-1])
    plt.ylabel('ToF [us]')
    showme()

def plot_thickness_models(models, labels, dt):
    """Plots different thickness models"""
    clf()
    colors_ = [0, -1]
    for i, model in enumerate(models):
        plt.plot(dt, model, label=labels[i], c=colors.cu_colors[colors_[i]])
    plt.legend(title='model')
    plt.title('Thickness Models')
    plt.xlabel('time [h]')
    showme()
