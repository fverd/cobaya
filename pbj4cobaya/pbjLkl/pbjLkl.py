r"""
.. module:: pbj_cobaya


"""

# Global
import importlib.util, sys
import numpy as np
# Local
from cobaya.log import LoggedError
from cobaya.likelihood import Likelihood

class pbjLkl(Likelihood):

    def initialize(self):
        if self.pbj_path is None:  # pragma: no cover
            raise LoggedError("No path to PBJ folder provided: define pbj_path please")
        sys.path.insert(0, self.pbj_path)
        try:
            # Import the module
            self.pbj = importlib.import_module('pbjcosmo')
        finally:
            # Remove the folder from sys.path after importing
            sys.path.pop(0)

        # Start building the PBJ init dictionary
        self.pbj_ini['input_cosmology']=None
        self.pbjobj = self.pbj.Pbj(self.pbj_ini)
        self.use_Pell = self.pbj_ini['likelihood']['observables']

        # one between kmax and NmaxP must be declared
        self.kF=2*np.pi/self.pbj_ini['boxsize']
        dk=self.pbj_ini['binning']['grid']['dk']
        cf=self.pbj_ini['binning']['grid']['cf']
        if 'kmax' in self.pbj_ini['binning']['grid']:
            self.k_max=self.pbj_ini['binning']['grid']['kmax']
            self.pbj_ini['binning']['grid']['nbinsP']=int((self.k_max/self.kF-cf)//dk)
        elif 'nbinsP' in self.pbj_ini['binning']['grid']:
            self.k_max=(dk*self.pbj_ini['binning']['grid']['nbinsP']+cf)*self.kF
        else:
            raise LoggedError(self.log, "One between kmax and NmaxP must be declared")
        self.datavector=np.loadtxt(self.data_path)

    def get_requirements(self):
        return {'Pk': {Pell: self.k_max for Pell in self.use_Pell}}
    
    def logp(self, **_params_values):
        Pkell = self.provider.get_Pk(_params_values)
        # return -0.5 * self.get_chi_squared(Pkell)
        return -0.5*100*(Pkell['P0']/self.datavector-1)**2
