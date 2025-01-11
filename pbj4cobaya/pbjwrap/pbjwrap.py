r"""
.. module:: pbj_cobaya


"""

# Global
import importlib.util, sys
import numpy as np
# Local
from cobaya.log import LoggedError
from cobaya.likelihood import Likelihood

class pbjwrap(Likelihood):

    def initialize(self):
        if self.pbj_path is None:  # pragma: no cover
            raise LoggedError("No path to PBJ folder provided: define pbj_path please")
        sys.path.insert(0, self.pbj_path)
        try:
            # Import the module
            self.pbj = importlib.import_module('pbjcosmo')
            self.pbj = importlib.reload(self.pbj)
        finally:
            # Remove the folder from sys.path after importing
            sys.path.pop(0)
        # Start building the PBJ init dictionary
        # one between kmax and NmaxP must be declared
        self.kF=2*np.pi/self.pbj_ini['boxsize']
        dk=self.pbj_ini['binning']['grid']['dk']
        cf=self.pbj_ini['binning']['grid']['cf']
        if self.k_max is not None:
            NPmax=(self.k_max/self.kF-cf)//dk+0.5
        else:
            raise LoggedError(self.log, "You must declare the k_max of the fit")

        # Finally call PBJ
        self.pbjobj = self.pbj.Pbj(self.pbj_ini)
        # Initialize evetything in PBJ but the likelihood part
        self.pbjobj.initialise_grid()
        self.pbjobj.initialise_data()
        self.pbjobj.CutVectors(NmaxP=NPmax, NminP=0, NmaxB=0, NmaxB2=None,
                        NmaxP4=None, NmaxB4=None)

        if 'model' in self.pbj_ini['likelihood'] and isinstance(self.pbj_ini['likelihood']['model'], str):
            try:
                self.pbjobj.model_function = getattr(self.pbjobj,
                                              self.pbj_ini['likelihood']['model'], None)
            except:
                raise NotImplementedError(r"Likelihood model function not implemented.")
        else:
            if self.pbj_ini['likelihood']['cosmology'] == 'fixed':
                if self.pbjobj.do_AP and not self.pbjobj.do_analytic_marg:
                    self.pbjobj.model_function = self.pbjobj.model_fixed_cosmology_AP
                elif self.pbjobj.do_analytic_marg:
                    self.pbjobj.model_function = self.pbjobj.model_fixed_cosmology_analytic_marg
                else:
                    self.pbjobj.model_function = self.pbjobj.model_fixed_cosmology

            elif self.pbj_ini['likelihood']['cosmology'] == 'varied':
                if self.pbjobj.do_analytic_marg:
                    self.pbjobj.model_function = self.pbjobj.model_varied_cosmology_analytic_marg
                else:
                    self.pbjobj.model_function = self.pbjobj.model_varied_cosmology
        print("\033[1;32m[info] \033[00m"+\
              f"Using {self.pbjobj.model_function.__name__} as likelihood model function")

        pbj_cosmo_pars=["h", "Obh2", "Och2", "ns", "As", "tau", "Tcmb", "z", "Mnu"]
        pbj_bias_pars=['b1']
        self.pbj_allpars = set(pbj_cosmo_pars) | set(pbj_bias_pars)
        self.pbjobj.varied_params = self.pbj_allpars
        self.pbjobj.full_param_dict = {key: 0. for key in self.pbj_allpars}
        self.pbjobj.prior_dictionary = {}
        self.renames_pbjtocob = {value: key for key, value in self.renames.items()}
        
    def get_requirements(self):
        requirements = []
        for k in self.pbj_allpars:
            if k in self.renames.values():
                for v in self.renames:
                    if self.renames[v] == k:
                        requirements.append((v, None))
                        break
            else:
                requirements.append((k, None))
        return requirements
  
    def translate_param(self, p):
        return self.renames_pbjtocob.get(p, p)
    
    def logp(self, **_params_values):
        parvals = [_params_values[self.translate_param(k)] for k in self.pbj_allpars]
        chi2r = self.pbjobj.model_function(parvals)
        # print(_params_values['b1'],self.pbjobj.TheoryVec[:2],np.hstack((self.pbjobj.CutDataVecs))[:,:2])
        lnL = -0.5*np.einsum('i->', chi2r)
        return lnL
