#!/bin/bash
#SBATCH -N 2
#SBATCH -q 
#SBATCH -J planck_cosmopower
#SBATCH -C haswell
#SBATCH -t 8:00:00

#OpenMP settings:
export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


conda activate base
conda list
cd /home/fverdian/cobaya

time srun -n 2 -c 16 --cpu_bind=cores cobaya-run -r FRA/planck_cosmopower.yaml > /home/fverdian/cobaya/scripts/planck_cosmopower.log 2>&1

wait