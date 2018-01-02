#!/usr/bin/env python3

from itertools import islice, product
import logging
import multiprocessing
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np

etamax = 5.1
deta = 0.1


def run_cmd(*args):
    """
    Run and log a subprocess.

    """
    cmd = ' '.join(args)
    logging.info('running command: %s', cmd)

    try:
        proc = subprocess.run(
            cmd.split(), check=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(
            'command failed with status %d:\n%s',
            e.returncode, e.output.strip('\n')
        )
        raise
    else:
        logging.debug(
            'command completed successfully:\n%s',
            proc.stdout.strip('\n')
        )
        return proc


def dnch_deta(event, eta_ranges):
    """
    Sample Nch within the finite pseudo-rapidity window
    eta_min < eta < eta_max according to a Poisson
    distribution.

    """
    nx, ny, neta = event.shape
    eta = np.linspace(-etamax, etamax, neta)

    entropy = sum([
        event[:, :, (eta_min <= eta) & (eta <= eta_max)].sum()
        for (eta_min, eta_max) in eta_ranges
    ])

    return np.random.poisson(entropy)


def trento(sys, norm=1, events='events.hdf'):
    """
    Yield minimum bias Trento3D events indefinitely
    Trento3D parameters are taken from table IV in reference,

    https://arxiv.org/pdf/1610.08490.pdf

    """
    xymax = {'Pb Pb': 10., 'p Pb': 6.}[sys]
    dxy = 0.5

    while True:
        try:
            os.remove(events)
        except FileNotFoundError:
            pass

        run_cmd(
            'trento3d', sys,
            '--number-events {}'.format(10**3),
            '--reduced-thickness {}'.format(0.0),
            '--fluctuation {}'.format(2.0),
            '--nucleon-width {}'.format(0.88),
            '--mean-coeff {}'.format(0.0),
            '--std-coeff {}'.format(2.9),
            '--skew-coeff {}'.format(7.3),
            '--jacobian {}'.format(0.75),
            '--xy-max {} --xy-step {}'.format(xymax, dxy),
            '--eta-max {} --eta-step {}'.format(etamax, deta),
            '--output', events,
        )

        with h5py.File(events, 'r') as f:
            for dset in f.values():
                mult = dset.attrs['mult']
                e2 = dset.attrs['e2']
                e3 = dset.attrs['e3']
                ev = np.array(dset)[:, :, ::-1]
                yield norm*ev, norm*mult, e2, e3


def write_attr(args):
    """
    Calculate events' attributes using each one of the
    experimental centrality estimators

    """
    # progress
    sys, batchid = args
    print(*args)

    # centrality estimators
    V0A = [(2.8, 5.1)]
    V0M = [(-3.7, -1.7), (2.8, 5.1)]
    CL1 = [(-1.4, 1.4)]
    estimators = (V0A, V0M, CL1)

    events_cache = Path('cache/trento/events_{}.hdf'.format(os.getpid()))
    if not events_cache.parent.exists():
        os.makedirs(events_cache.parent, exist_ok=True)

    trento_events = trento(sys, norm=0.265, events=str(events_cache))
    attr = []

    for (ev, mult, e2, e3) in islice(trento_events, 10**3):
        nch_est = [dnch_deta(ev, eta_ranges) for eta_ranges in estimators]
        nch_mid = dnch_deta(ev, [(-.5, .5)])
        attr.append([*nch_est, nch_mid, mult, e2, e3])

    filename = 'cache/trento/{}.dat'.format(sys.replace(' ', ''))
    with open(filename, 'ab') as f:
        np.savetxt(f, np.array(attr))

    os.remove(events_cache)


def main():

    # trento parameters
    systems = ('p Pb', 'Pb Pb')
    repeat = 10

    # initial multiple processes
    multiprocessing.Pool(4).map(
        write_attr, product(systems, range(repeat))
        )


if __name__ == "__main__":
    main()
