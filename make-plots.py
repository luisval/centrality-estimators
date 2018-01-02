#!/usr/bin/env python3

from collections import defaultdict
import logging
import pickle
import re
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import yaml


# ALICE centrality estimators
estimators = ('V0A', 'V0M', 'CL1')
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()
color = dict(zip(estimators, color_cycle['color']))


class HEPData:
    """
    Interface to a HEPData yaml file.

    Downloads and caches the dataset specified by the INSPIRE record and table
    number.  The web UI for `inspire_rec` may be found at:

        https://hepdata.net/record/ins`inspire_rec`

    If `reverse` is true, reverse the order of the data table (useful for
    tables that are given as a function of Npart).

    """
    def __init__(self, inspire_rec, table, reverse=False):
        cachedir = Path('cache')
        cachefile = (
            cachedir / 'hepdata' /
            'ins{}_table{}.pkl'.format(inspire_rec, table)
        )
        name = 'record {} table {}'.format(inspire_rec, table)

        if cachefile.exists():
            logging.debug('loading from hepdata cache: %s', name)
            with cachefile.open('rb') as f:
                self._data = pickle.load(f)
        else:
            logging.debug('downloading from hepdata.net: %s', name)
            cachefile.parent.mkdir(exist_ok=True)
            with cachefile.open('wb') as f, urlopen(
                    'https://hepdata.net/download/table/'
                    'ins{}/Table{}/yaml'.format(inspire_rec, table)
            ) as u:
                self._data = yaml.load(u)
                pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if reverse:
            for v in self._data.values():
                for d in v:
                    d['values'].reverse()

    @property
    def names(self):
        """
        Get the independent variable names.

        """
        data = self._data['dependent_variables']

        return [d['header']['name'] for d in data]

    def x(self, name, case=True):
        """
        Get an independent variable ("x" data) with the given name.

        If `case` is false, perform case-insensitive matching for the name.

        """
        trans = (lambda x: x) if case else (lambda x: x.casefold())
        name = trans(name)

        for x in self._data['independent_variables']:
            if trans(x['header']['name']) == name:
                return x['values']

        raise LookupError("no x data with name '{}'".format(name))

    @property
    def cent(self):
        """
        Return a list of centrality bins as (low, high) tuples.

        """
        try:
            return self._cent
        except AttributeError:
            pass

        x = self.x('centrality', case=False)

        if x is None:
            raise LookupError('no centrality data')

        try:
            cent = [(v['low'], v['high']) for v in x]
        except KeyError:
            # try to guess bins from midpoints
            mids = [v['value'] for v in x]
            width = set(a - b for a, b in zip(mids[1:], mids[:-1]))
            if len(width) > 1:
                raise RuntimeError('variable bin widths')
            d = width.pop() / 2
            cent = [(m - d, m + d) for m in mids]

        self._cent = cent

        return cent

    @cent.setter
    def cent(self, value):
        """
        Manually set centrality bins.

        """
        self._cent = value

    def y(self, name=None, **quals):
        """
        Get a dependent variable ("y" data) with the given name and qualifiers.

        """
        for y in self._data['dependent_variables']:
            if name is None or y['header']['name'] == name:
                y_quals = {q['name']: q['value'] for q in y['qualifiers']}
                if all(y_quals[k] == v for k, v in quals.items()):
                    return y['values']

        raise LookupError(
            "no y data with name '{}' and qualifiers '{}'"
            .format(name, quals)
        )

    def dataset(self, name=None, maxcent=100, ignore_bins=[], **quals):
        """
        Return a dict containing:

            cent : list of centrality bins
            x : np.array of centrality bin midpoints
            y : np.array of y values
            yerr : subdict of np.arrays of y errors

        `name` and `quals` are passed to HEPData.y()

        Missing y values are skipped.

        Centrality bins whose upper edge is greater than `maxcent` are skipped.

        Centrality bins in `ignore_bins` [a list of (low, high) tuples] are
        skipped.

        """
        cent = []
        y = []
        yerr = defaultdict(list)

        for c, v in zip(self.cent, self.y(name, **quals)):
            # skip missing values
            # skip bins whose upper edge is greater than maxcent
            # skip explicitly ignored bins
            if v['value'] == '-' or c[1] > maxcent or c in ignore_bins:
                continue

            cent.append(c)
            y.append(v['value'])

            for err in v['errors']:
                try:
                    e = err['symerror']
                except KeyError:
                    e = err['asymerror']
                    if abs(e['plus']) != abs(e['minus']):
                        raise RuntimeError(
                            'asymmetric errors are not implemented'
                        )
                    e = abs(e['plus'])

                yerr[err.get('label', 'sum')].append(e)

        return dict(
            cent=cent,
            x=np.array([(a + b)/2 for a, b in cent]),
            y=np.array(y),
            yerr={k: np.array(v) for k, v in yerr.items()},
        )


def ALICE_pPb5020(ax, est):
    """
    ALICE p+Pb 5020 TeV dNch/deta

    p going side: eta < 0
    Pb going side: eta > 0
    eta_beam = -.465

    eta_cms = eta_lab - eta_beam
    i.e. eta_cms = eta_lab + .465

    Thus if we want
    -0.5 < eta_cms < 0.5, then
    -0.5 < eta_lab - eta_beam < 0.5, and
    -0.5 + eta_beam < eta_lab < 0.5 + eta_beam

    """
    tbl = {'CL1': 1, 'V0M': 2, 'V0A': 3}[est]
    eta_beam = -0.465
    eta_cut = 1.4

    # use the CL1 centrality estimator with |eta| < 1.4
    dset = HEPData(1335350, tbl)

    cent = [tuple(map(float, re.findall(r'\d+', name)))
            for name in dset.names]

    eta_lab_min, eta_lab_max = [eta + eta_beam for eta in (-eta_cut, eta_cut)]

    y, stat, sys = np.array([[
        (y['value'], y['errors'][0]['symerror'], y['errors'][1]['symerror'])
        for (x, y) in zip(dset.x('$\eta_{lab}$'), dset.y(name))
        if eta_lab_min < x['low'] and x['high'] < eta_lab_max
    ] for name in dset.names]).T

    x = np.array([(a + b)/2 for a, b in cent])
    y = y.mean(axis=0)
    stat = np.sqrt(np.square(stat).sum(axis=0))/len(stat)
    sys = sys.mean(axis=0)

    ax.errorbar(
        x, y, yerr=np.sqrt(stat**2 + sys**2),
        fmt='o', label='ALICE {}'.format(est), color=color[est]
    )

    ax.set_xlabel('Centrality %')
    ax.set_ylabel(r'$dN_\mathrm{ch}/d\eta$')
    ax.set_title('p+Pb 5.02 TeV')
    ax.legend()


def ALICE_PbPb5020(ax):
    """
    ALICE Pb+Pb 5020 TeV dNch/deta

    """

    name = r'$\mathrm{d}N_\mathrm{ch}/\mathrm{d}\eta$'
    dset = HEPData(1410589, 2).dataset(name)

    x = dset['x']
    y = dset['y']
    yerr = np.sqrt(
        dset['yerr']['stat']**2 + dset['yerr']['sys']**2
    )

    ax.errorbar(
        x, y, yerr=yerr, fmt='o',
        label='ALICE V0M', color=color['V0M']
    )

    ax.set_xlabel('Centrality %')
    ax.set_ylabel(r'$dN_\mathrm{ch}/d\eta$')
    ax.set_title('Pb+Pb 5.02 TeV')
    ax.legend()


def TRENTO(ax, system):
    """
    Calculate dNch/deta as a function of collision centrality
    using initial entropy scaling.

    """

    *nch_estimators, nch_midrapidity = np.loadtxt(
            'cache/trento/{}.dat'.format(system), usecols=(0, 1, 2, 3)
            ).T

    x = np.linspace(0, 100, 20)

    for nch_est, est in zip(nch_estimators, estimators):
        sorted_indices = nch_est.argsort()
        y = nch_midrapidity[sorted_indices[::-1]].reshape(20, -1).mean(axis=1)
        ax.plot(x, y, color=color[est])


def main():
    """
    Plot and compare different experimental methods for estimating
    centrality in p+Pb and Pb+Pb collisions at 5.02 TeV.

    """
    # figure dimensions
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    ax1, ax2 = axes

    # collision systems and axes
    systems = ('pPb', 'PbPb')

    # plot experiment
    for est in estimators:
        ALICE_pPb5020(ax1, est)

    ALICE_PbPb5020(ax2)

    # plot trento3d calculations
    for (ax, sys) in zip(axes, systems):
        try:
            TRENTO(ax, sys)
        except FileNotFoundError:
            pass

    plt.tight_layout()
    #for ax in axes:
    #    ax.set_yscale('log')
    plt.savefig('yields.pdf')


if __name__ == "__main__":
    main()
