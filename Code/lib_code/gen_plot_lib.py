import os
from os import listdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
sns.set_style("white")
CLR = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'] + plt.rcParams['axes.prop_cycle'].by_key()['color'])
EPS = 10**-16

cmap = 'YlOrRd'
# cmap_d = 'RdBu_r'
cmap_d = 'seismic'

# SMALL_SIZE = 13
# MEDIUM_SIZE = 15
# BIGGER_SIZE = 17
#
# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def define_font_size(small_size=13, medium_size=15, bigger_size=17):
    plt.rc('font', size=medium_size)          # controls default text sizes
    plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=small_size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)    # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

define_font_size()

def get_clustermap_idx(cm, df=None):
    # Returns the indices of clustermap as a dict{'row': x, 'col': y},
    #       either as numbers or by the index and columns of df (if provided).

    if df is not None: return {'row': df.index[cm.dendrogram_row.reordered_ind], 'col': df.columns[cm.dendrogram_col.reordered_ind]}
    return {'row': cm.dendrogram_row.reordered_ind, 'col': cm.dendrogram_col.reordered_ind}

def scatter(a, b, log=False, log10=False, diag=True, ma=0, ax=None, c='black', s=1,
            label=None, alpha=1, aspect=False, cmap=None, return_scat_obj=False, args={}):
    # Scatters a vs b of ax (default plt). Can apply:
    #   -   log(x+1) transformation (optional)
    #   -   label (optional)
    #   -   color (default black)
    #   -   size of point (default 1)
    #   -   alpha factor (default 1)
    #   -   aspect ratio equal (optional)
    #   -   diagonal line from 0 to ma (if provided) or to max(a,b) (optional)

    ax_ = ax if ax is not None else plt
    if log:
        a = a+1
        b = b+1
        ax_.semilogx(base=2)
        ax_.semilogy(base=2)
    elif log10:
        a = a+1
        b = b+1
        ax_.semilogx(base=10)
        ax_.semilogy(base=10)
    if diag:
        if ma == 0: ma = np.max([a.max(), b.max()])
        ax_.plot([0, ma], [0, ma], linewidth=1, c='black')
    scat = ax_.scatter(a, b, s=s, c=c, alpha=alpha, label=label, cmap=cmap, **args)
    if aspect:
        if ax is None: ax_.gca().set_aspect('equal')
        else: ax_.set_aspect('equal')
    if return_scat_obj: return ax_, scat
    else: return ax_

def add_labels(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
               xticks=None, xticklabels=None, xtickrotation=None,
               yticks=None, yticklabels=None, ytickrotation=None,
               aspect=False, legend=False, ax=None):
    # Applies labels (if provided)

    if ax is None or ax is plt:
        if title is not None: plt.title(title)
        if xlabel is not None: plt.xlabel(xlabel)
        if ylabel is not None: plt.ylabel(ylabel)
        if xlim is not None: plt.xlim(xlim)
        if ylim is not None: plt.ylim(ylim)
        if xticks is not None:
            if xtickrotation is not None: plt.xticks(xticks, xticklabels, rotation=xtickrotation)
            else: plt.xticks(xticks, xticklabels)
        if yticks is not None:
            if ytickrotation is not None: plt.yticks(yticks, yticklabels, rotation=ytickrotation)
            else: plt.yticks(yticks, yticklabels)
        if aspect: plt.gca().set_aspect('equal')
        if legend: plt.legend()
        return plt
    else:
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if xticks is not None: ax.set_xticks(xticks)
        if xticklabels is not None:
            if xtickrotation is not None: ax.set_xticklabels(xticklabels, rotation=xtickrotation)
            else: ax.set_xticklabels(xticklabels)
        if yticks is not None:
            if ytickrotation is not None: ax.set_yticklabels(yticklabels, rotation=ytickrotation)
            else: ax.set_yticklabels(yticklabels)
        if yticklabels is not None: ax.set_yticklabels(yticklabels)
        if aspect: ax.set_aspect('equal')
        if legend: ax.legend()
        return ax

def add_log10_ticks(ax=None, diag=False):
    if ax is None or ax is plt: ax = plt.gca()
    xma, yma = ax.get_xlim()[-1], ax.get_ylim()[-1]

    ticklabels = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    ticks = ticklabels + 1
    xidx = (np.abs(xma - ticks)).argmin()
    yidx = (np.abs(yma - ticks)).argmin()
    xticks = ticks[:xidx+1]
    xticklabels = ticklabels[:xidx+1]
    yticks = ticks[:yidx+1]
    yticklabels = ticklabels[:yidx+1]

    if diag:
        ma = np.max([yticks[-1], xticks[-1]])
        ax.plot([0, ma], [0, ma], linewidth=1, c='black')

    add_labels(xticks=xticks, xticklabels=xticklabels,
                   yticks=yticks, yticklabels=yticklabels, ax=ax)


def show(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, aspect=False,
         fig=plt, figname=None, tight_layout=True, dpi=300,
         savefig=True, showfig=True, legend=False, savepdf=False):
    # Finishes plot:
    #   -   Applies labels (if provided)
    #   -   Applies legend and tight_layout (optional)
    #   -   Saves to figname (optional)
    #   -   Shows fig (optional - otherwise delete it)

    if type(fig) is plt.Figure and title is not None: fig.suptitle(title)
    if type(fig) is not plt.Figure:
        add_labels(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, aspect=aspect, ax=None)
    else:
        add_labels(xlabel=xlabel, ylabel=ylabel, aspect=aspect, ax=None)

    if legend: plt.legend()
    if tight_layout: fig.tight_layout()
    if savefig and figname is not None: fig.savefig(figname.parent/f'{figname.name}.png', dpi=dpi)
    if savepdf and figname is not None: fig.savefig(figname.parent/f'{figname.name}.pdf')
    if showfig: plt.show()
    else: fig.clf()


def add_legend_patches(labels, colors, ax=plt, set_legend=True):
    # Sets patches for legend in ax, matches label to color by order of lists
    patches = []
    for l, c in zip(labels, colors):
        patches.append(mpatches.Patch(color=c, label=l))
    if set_legend: ax.legend(handles=patches)
    return patches


def define_log_ticklabels(ax, x_ticks=None, y_ticks=None, set_x=True, set_y=True):
    if set_x:
        ax.set_xticks([0] + [np.log2(2 ** k + 1) for k in x_ticks])
        ax.set_xticklabels(['0'] + [rf'${k}$' for k in ['2^{' + f'{k}' + '}' for k in x_ticks]])
    if set_y:
        ax.set_yticks([0] + [np.log2(2 ** k + 1) for k in y_ticks])
        ax.set_yticklabels(['0'] + [rf'${k}$' for k in ['2^{' + f'{k}' + '}' for k in y_ticks]])
