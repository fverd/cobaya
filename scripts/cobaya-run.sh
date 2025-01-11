#!/bin/bash
#SBATCH -N 2
#SBATCH -q 
#SBATCH -J cobaya-run
#SBATCH -C haswell
#SBATCH -t 8:00:00

#OpenMP settings:
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


conda activate base
conda list
cd /home/fverdian/cobaya

cobaya-run time srun -n 2 -c 8 --cpu_bind=cores cobaya-run -r cobaya-run.yaml > /home/fverdian/cobaya/scripts/cobaya-run.log 2>&1
time srun -n 2 -c 8 --cpu_bind=cores cobaya-run -r FRA/planck_cosmopower.yaml > /home/fverdian/cobaya/scripts/planck_cosmopower.log 2>&1

wait