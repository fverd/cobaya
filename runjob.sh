#!/usr/bin/env bash
#
#SBATCH --job-name=boss_k0p05

#SBATCH --mail-type=END
#SBATCH --mail-user=fverdian@sissa.it
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=46G

#SBATCH --time=48:00:00  
#
#SBATCH --partition=batch
#SBATCH --output=Slurm-output/%x.o%j
export OMP_NUM_THREADS=$((${SLURM_CPUS_PER_TASK}/2))
export RDMAV_FORK_SAFE=1

cobaya-run FRA-params/fx_boss.yaml -o chains-fx/kJ0p05/chain -f