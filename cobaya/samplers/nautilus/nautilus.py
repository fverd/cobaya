"""
.. module:: samplers.nautilus

:Synopsis: Interface for the Nautilus sampler
:Author: fv
"""
# Global
import os
import sys
import logging
import inspect
from itertools import chain
from typing import Any, Callable, Union, Dict, Optional
from tempfile import gettempdir
import re
import warnings
import numpy as np
from scipy.stats import norm

# Local
from cobaya.tools import read_dnumber, get_external_function, find_with_regexp, \
    NumberWithUnits, get_compiled_import_path
from cobaya.sampler import Sampler
from cobaya.mpi import is_main_process, share_mpi, sync_processes
from cobaya.collection import SampleCollection
from cobaya.log import get_logger, NoLogging, LoggedError
from cobaya.install import download_github_release
from cobaya.component import ComponentNotInstalledError, load_external_module
from cobaya.yaml import yaml_dump_file
from cobaya.conventions import derived_par_name_separator, Extension

# Suppresses warnings about first defining attrs outside __init__
# pylint: disable=attribute-defined-outside-init


class nautilus(Sampler):

    # variables from yaml
    n_live: int
    n_eff: int
    poolN: int
    logzero: float

    def initialize(self):
        """Imports the Nautilus sampler and prepares its arguments."""
        try:
            self.naut = load_external_module("nautilus")
        except ComponentNotInstalledError as excpt:
            raise ComponentNotInstalledError(
                self.log, (f"Could not find nautilus: {excpt}. ")) from excpt
        # Prepare arguments and settings
        self.n_sampled = len(self.model.parameterization.sampled_params())
        self.n_derived = len(self.model.parameterization.derived_params())
        self.n_priors = len(self.model.prior)
        self.n_likes = len(self.model.likelihood)
        self.nDims = self.model.prior.d()
        self.nDerived = self.n_derived + self.n_priors + self.n_likes
        if self.logzero is None:
            self.logzero = np.nan_to_num(-np.inf)

        # Prepare output folders and prefixes
        if self.output:
            self.file_root = self.output.prefix
            self.read_resume = self.output.is_resuming()
        else:
            output_prefix = share_mpi(hex(int(self._rng.random() * 16 ** 6))[2:]
                                      if is_main_process() else None)
            self.file_root = output_prefix
            # dummy output -- no resume!
            self.read_resume = False

        sampled_params_info = self.model.prior._parameterization.sampled_params_info()

        # PREPARE THE PRIOR
        # As for the polychord sampler, I sample directly the posterior as if it was the likelihood
        # Thi is described in https://nautilus-sampler.readthedocs.io/en/latest/guides/priors.html "without transformations"
        # Anyway, we must have an integration cube to give to the nautilus sampler 
        # For the moment, for Gaussian priors I use a uniform interva of 4 sigmas
        self.naut_flat_prior = self.naut.Prior()
        for p in self.model.parameterization.sampled_params():
            if(sampled_params_info[p].get("prior").get("dist"))=='norm':
                ploc=sampled_params_info[p].get("prior").get('loc')
                pscale=sampled_params_info[p].get("prior").get('scale')
                self.mpi_info(f'Wide flat prior of 3sigma for {p}: (mu={ploc}, sigma={pscale})')
                self.naut_flat_prior.add_parameter(p, dist=(ploc-3*pscale, ploc+3*pscale))
            else:
                pmin=sampled_params_info[p].get("prior").get('min')
                pmax=sampled_params_info[p].get("prior").get('max')
                self.mpi_info(f'Flat prior on {p}: [{pmin},{pmax}]')
                self.naut_flat_prior.add_parameter(p, dist=(pmin, pmax))

        # Done!
        if is_main_process():
            self.log.debug("Nautilus loaded correctly:")
        self.logZ, self.logZstd = np.nan, np.nan

    def run(self):
        """
        Prepares the prior and likelihood functions, calls ``Nautilus``'s ``run``, and
        processes its output.
        """

        def logpost(params_values):
            result = self.model.logposterior(params_values)
            loglikes = result.loglikes
            if len(loglikes) != self.n_likes:
                loglikes = np.full(self.n_likes, np.nan)
            derived = result.derived
            if len(derived) != self.n_derived:
                derived = np.full(self.n_derived, np.nan)
            derived = list(derived) + list(result.logpriors) + list(loglikes)
            if result.logpost != result.logpost:
                return self.logzero, derived
            return max(result.logpost, self.logzero), derived
        
        sync_processes()
        self.mpi_info("Calling Nautilus...")

        self.naut_sampler = self.naut.Sampler(self.naut_flat_prior, logpost, n_live=self.n_live, pool=self.poolN, pass_dict=False, filepath=self.checkpoint_filename())
        self.naut_sampler.run(verbose=True, discard_exploration=False, n_eff=self.n_eff)
        self.process_raw_output()

    def dump_paramnames(self, prefix):
        labels = self.model.parameterization.labels()
        with open(prefix + ".paramnames", "w", encoding="utf-8-sig") as f_paramnames:
            for p in self.model.parameterization.sampled_params():
                f_paramnames.write("%s\t%s\n" % (p, labels.get(p, "")))
            for p in self.model.parameterization.derived_params():
                f_paramnames.write("%s*\t%s\n" % (p, labels.get(p, "")))
            for p in self.model.prior:
                f_paramnames.write("%s*\t%s\n" % (
                    "logprior" + derived_par_name_separator + p,
                    r"\log\pi_\mathrm{" + p.replace("_", r"\ ") + r"}"))
            for p in self.model.likelihood:
                f_paramnames.write("%s*\t%s\n" % (
                    "loglike" + derived_par_name_separator + p,
                    r"\log\mathcal{L}_\mathrm{" + p.replace("_", r"\ ") + r"}"))

    def save_sample(self, fname, name):
        collection = SampleCollection(
            self.model, self.output, name=str(name), sample_type="nested")
        points, log_w, log_l, derpoints= self.naut_sampler.posterior(equal_weight=True, return_blobs=True)
        for i in range(points.shape[0]):
            collection.add(points[i], logpost=float(log_l[i]), weight=np.exp(log_w[i]),
                           derived=derpoints[i,:self.n_derived],
                           logpriors=derpoints[i,-(self.n_priors + self.n_likes):-self.n_likes],
                           loglikes=derpoints[i,-self.n_likes:],
                           )
        # make sure that the points are written
        collection.out_update()
        return collection

    def process_raw_output(self):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if not is_main_process():
            return
        self.dump_paramnames(self.raw_prefix)
        self.collection = self.save_sample(self.raw_prefix + ".txt", "1")
        self.log.info("Removing checkpoint file '%s'", self.checkpoint_filename())
        os.remove(self.checkpoint_filename())
        self.log.info("Finished! Nautilus output stored in '%s'", self.raw_prefix)

    def samples(
            self,
            combined: bool = False,
            skip_samples: float = 0,
            to_getdist: bool = False,
    ) -> Union[SampleCollection, "MCSamples"]:
        """
        Returns the sample of the posterior built out of dead points.

        Parameters
        ----------
        combined: bool, default: False
            If ``True`` returns the same, single posterior for all processes. Otherwise,
            it is only returned for the root process (this behaviour is kept for
            compatibility with the equivalent function for MCMC).
        skip_samples: int or float, default: 0
            No effect (skipping initial samples from a sorted nested sampling sample would
            bias it). Raises a warning if greater than 0.
        to_getdist: bool, default: False
            If ``True``, returns a single :class:`getdist.MCSamples` instance, containing
            all samples, for all MPI processes (``combined`` is ignored).

        Returns
        -------
        SampleCollection, getdist.MCSamples
           The posterior sample.
        """
        if skip_samples:
            self.mpi_warning(
                "Initial samples should not be skipped in nested sampling. "
                "Ignoring 'skip_samples' keyword."
            )
        collection = self.collection
        if not combined and not to_getdist:
            return collection  # None for MPI ranks > 0
        # In all remaining cases, we return the same for all ranks
        if to_getdist:
            if is_main_process():
                collection = collection.to_getdist()
        return share_mpi(collection)

    def checkpoint_filename(self):
        if self.output:
            return os.path.join(
                self.output.folder, self.output.prefix + '-state.hdf5')
        return None
    @property
    def raw_prefix(self):
        return os.path.join(self.output.folder, self.output.prefix)
        