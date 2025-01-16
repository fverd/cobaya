import pbjcosmo 
import pbjcosmo.tools.param_handler as parhandler
import matplotlib.pyplot as plt
import numpy as np

# Load the parameter file and initialise a Pbj object
init_config = parhandler.read_file("param_theory.yaml")
theoryobject = pbjcosmo.Pbj(init_config)

# Compute the multipoles of the power spectrum

# Set a k-grid on which the model will be evaluated
kvals = np.linspace(0.001, 0.8, 350)

# Call pbj.P_kmu_z() to get the multipoles. You must specify a
# redshift, if you want to do redshift rescaling (i.e., do you want
# to rescale from a linear power spectrum computed at z=0 using the
# linear growth factor, or do you want to start from a linear power
# spectrum computed at the given redshift?). Optional parameters
# include: the grid of k values on which the model is evaluated, a
# value for the growth functions f,D, a dictionary containing the
# cosmology (if not specified, the one in tha parameterfile is used),
# if you want to apply Alcock-Packinsky distortions by giving values
# for the alphas that are not computed from the cosmology, values for
# the bias, counterterms and noise parameters. Psn is 1/nbar, if not
# specified the theory prediction will not include shot noise
p0, p2, p4 = theoryobject.P_kmu_z(1.0, True, kgrid=kvals, b1=1.5, b2=0.7, bG2=0.3,
                                  bG3=-0.7, c0=5, c2=10., c4=20., ck4=-5., aP=0.5,
                                  e0k2=5, e2k2=10, Psn=5e-3)

# Plot the multipoles
plt.loglog(kvals, p0, label=r'$P_0$')
plt.loglog(kvals, p2, label=r'$P_2$')
plt.loglog(kvals, p4, label=r'$P_4$')
plt.legend()
plt.xlabel('$k$')
plt.ylabel('$P_{\ell}(k)$')
plt.show()
