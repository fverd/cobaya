"""
.. module:: BoltzmannBase

:Synopsis: Template for Cosmological theory codes.
           Mostly here to document how to compute and get observables.
:Author: Jesus Torrado

"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Mapping, Iterable, Tuple
import importlib.util, sys

# Local
from cobaya.theories.cosmo import BoltzmannBase
from cobaya.theory import Theory
from cobaya.tools import deepcopy_where_possible, combine_1d, combine_2d, check_2d
from cobaya.log import LoggedError, abstract, get_logger

class PbjTheo(Theory):

    def initialize(self):
        self.kmax = 0
        pbj_cosmo_pars=["h", "Obh2", "Och2", "ns", "As", "tau", "Tcmb", "z", "Mnu"]
        pbj_bias_pars=['b1']
        self.allpars= set(pbj_cosmo_pars) | set(pbj_bias_pars)

        if self.pbj_path is None:
            raise LoggedError("No path to PBJ folder provided: define pbj_path please")
        sys.path.insert(0, self.pbj_path)
        try:
            # Import the module
            self.pbj = importlib.import_module('pbjcosmo')
        finally:
            # Remove the folder from sys.path after importing
            sys.path.pop(0)
        # Load the parameter file, initialise a PBJ object and run init_theory
        # init_config = self.pbj.tools.param_handler.read_file("/home/fverdian/cobaya/FRA/param_pbj.yaml")
        # self.pbjtheoryobject = self.pbj.Pbj(init_config)
        
    def get_requirements(self) -> Iterable[Tuple[str, str]]:
        requirements = []
        for k in self.allpars:
            if k in self.renames.values():
                for v in self.renames:
                    if self.renames[v] == k:
                        requirements.append((v, None))
                        break
            else:
                requirements.append((k, None))
        return requirements
    
    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider
        print(provider.__dir__)


    def must_provide(self, **requirements):
        r"""
        Specifies the quantities that this Boltzmann code is requested to compute.
        """
        # Accumulate the requirements across several calls in a safe way;
        # e.g. take maximum of all values of a requested precision parameter
        for k in requirements:
            # Products and other computations
            if k == 'Pk':
                # arguments are all identical, collect all in Pk_grid
                pass
            # Extra derived parameters and other unknown stuff (keep capitalization)
            else:
                raise LoggedError(self.log, "Unknown required product: '%s'.", k)
    
    def calculate(self, state: dict, want_derived: bool = True, **params) -> bool:
        # Compute the multipoles of the power spectrum
        # Set a k-grid on which the model will be evaluated
        kvals = np.linspace(0.01, 0.8, 50)
        p0, p2, p4 = self.pbjobj.P_kmu_z(1.0, True, kgrid=kvals, b1=params['b1'], b2=0.7, bG2=0.3,
                                        bG3=-0.7, c0=5, c2=10., c4=20., ck4=-5., aP=0.5,
                                        e0k2=5, e2k2=10, Psn=5e-3)
        # Pell_eff = self.P_kmu_z(self.z, self.do_redshift_rescaling,
        #                                         cosmo=self.full_param_dict, Psn=self.Psn,
        #                                         kgrid=self.kPE, **self.full_param_dict)
        Ps = {"k": kvals[:4]}
        state['P0']=p0[:4]

        return True

    def get_Pk(self, nonlinear=True, **params):
        r"""

        """
        return self.current_state.copy()