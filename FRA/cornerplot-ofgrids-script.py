import getdist.plots as gdplt
import os, fnmatch
from cobaya import load_samples
import numpy as np
import matplotlib.pyplot as plt

def loadgridchains(path, excludec=[]):
    Samples={}
    for root, dirs, files in os.walk(path):
        for chain_name in sorted(fnmatch.filter(dirs, 'base_*')):
            if chain_name in excludec: continue
            subdirs = next(os.walk(os.path.join(root, chain_name)))[1]
            only_subdir = subdirs[0]
            subdir_path = os.path.join(root, chain_name, only_subdir)
            Samples[chain_name] = load_samples(subdir_path+'/'+chain_name+'_'+only_subdir, to_getdist=True)
            Samples[chain_name].label = chain_name
            Samples[chain_name].ma_val = -int(chain_name[-2:])
            p=Samples[chain_name].getParams()
            Samples[chain_name].addDerived(p.omega_scf/(p.omega_cdm+p.omega_b+p.omega_scf), name='fx', label=r'f_\chi', range=[0.,None])
    return list(Samples.values())

grid_root_dir = '/home/fverdian/cobaya/chains-ulysses/axigrid_cmblss_nobr'
chains_to_plot=loadgridchains(grid_root_dir)

plotsdir = '/home/fverdian/cobaya/FRA/cornerplots/Planck+BOSS/'
os.makedirs(plotsdir, exist_ok=True)

for chain in chains_to_plot:

    print(f'Plotting {chain.ma_val}', flush=True)
    parkeys = [pn.name for pn in chain.paramNames.names]
    excludepars = ['Omega_scf', 'm_axion', 'kJ0p5', 'A_s', 'fx']
    pars_toplot = [s for s in parkeys if not (s.startswith('chi2') or s.startswith('minus') or s in excludepars)]

    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(chain , pars_toplot, title_limit=1, contour_colors=['purple'], filled=True)
    commentstring = r'Planck + BOSS ($P_\ell + Q_0$), ' + r'$m_a=10^{'+str(chain.ma_val)+r'}$'
    gdplot.fig.text(0.5, 0.9, commentstring, ha='center', va='center', fontsize=18)
    gdplot.finish_plot(no_tight=True)
    gdplot.fig.savefig(plotsdir+'ma'+str(chain.ma_val)+'.pdf', bbox_inches='tight')